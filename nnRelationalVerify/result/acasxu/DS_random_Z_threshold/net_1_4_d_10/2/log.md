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
execution time: IAR + RelationalAnalysis = 0.82 + 0.80 = 1.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0559989, upper bound: 0.0559989

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556532, upper bound: 0.0559050
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556532, upper bound: 0.0556532
time: 0.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.45 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0556532, upper bound: 0.0559050
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0556532, upper bound: 0.0556532

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554753, upper bound: 0.0557374
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554753, upper bound: 0.0558888
time: 0.22 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558813, upper bound: 0.0555895
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555895, upper bound: 0.0556385
time: 0.21 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.31 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -0.0554753, upper bound: 0.0557374
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -0.0554753, upper bound: 0.0558888
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -0.0558813, upper bound: 0.0555895
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.31
Output dim: 0, lower bound: -0.0555895, upper bound: 0.0556385

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553460, upper bound: 0.0556600
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553460, upper bound: 0.0556000
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553837, upper bound: 0.0556987
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553837, upper bound: 0.0554808
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553748, upper bound: 0.0554517
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553748, upper bound: 0.0553985
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554829, upper bound: 0.0554829
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554829, upper bound: 0.0556385
time: 0.24 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.34 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.34
Output dim: 0, lower bound: -0.0553460, upper bound: 0.0556600
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.34
Output dim: 0, lower bound: -0.0553460, upper bound: 0.0556000
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.34
Output dim: 0, lower bound: -0.0553837, upper bound: 0.0556987
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.34
Output dim: 0, lower bound: -0.0553837, upper bound: 0.0554808
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.34
Output dim: 0, lower bound: -0.0553748, upper bound: 0.0554517
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.34
Output dim: 0, lower bound: -0.0553748, upper bound: 0.0553985
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.34
Output dim: 0, lower bound: -0.0554829, upper bound: 0.0554829
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.34
Output dim: 0, lower bound: -0.0554829, upper bound: 0.0556385

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556088, upper bound: 0.0556600
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553866, upper bound: 0.0556548
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552454, upper bound: 0.0554578
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552476, upper bound: 0.0552458
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553837, upper bound: 0.0556987
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553786, upper bound: 0.0556984
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0553543
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551879, upper bound: 0.0552459
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552350, upper bound: 0.0552594
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552325, upper bound: 0.0553316
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556942, upper bound: 0.0553985
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554705, upper bound: 0.0553272
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558535, upper bound: 0.0554453
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557114, upper bound: 0.0554646
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555232, upper bound: 0.0547335
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556819, upper bound: 0.0556006
time: 0.21 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0556088, upper bound: 0.0556600
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0553866, upper bound: 0.0556548
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0552454, upper bound: 0.0554578
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0552476, upper bound: 0.0552458
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0553837, upper bound: 0.0556987
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0553786, upper bound: 0.0556984
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0553543
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0551879, upper bound: 0.0552459
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0552350, upper bound: 0.0552594
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0552325, upper bound: 0.0553316
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0556942, upper bound: 0.0553985
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0554705, upper bound: 0.0553272
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0558535, upper bound: 0.0554453
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0557114, upper bound: 0.0554646
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0555232, upper bound: 0.0547335
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.22
Output dim: 0, lower bound: -0.0556819, upper bound: 0.0556006

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549348, upper bound: 0.0549297
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550777, upper bound: 0.0551111
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552443, upper bound: 0.0554759
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552773, upper bound: 0.0549327
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0553490
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552454, upper bound: 0.0554578
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0552222
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552339, upper bound: 0.0552118
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549839, upper bound: 0.0554704
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549760, upper bound: 0.0550693
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549864, upper bound: 0.0554409
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549864, upper bound: 0.0553811
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552469, upper bound: 0.0553543
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552468, upper bound: 0.0547019
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545979, upper bound: 0.0552446
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551824, upper bound: 0.0548004
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550562, upper bound: 0.0551621
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552118, upper bound: 0.0552339
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552315, upper bound: 0.0552797
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551623, upper bound: 0.0552787
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556942, upper bound: 0.0553809
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556929, upper bound: 0.0553985
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554200, upper bound: 0.0547311
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554407, upper bound: 0.0553030
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558089, upper bound: 0.0552954
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557230, upper bound: 0.0553423
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556211, upper bound: 0.0554509
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548246, upper bound: 0.0547333
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550321, upper bound: 0.0542705
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550036, upper bound: 0.0542718
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555004, upper bound: 0.0553922
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551182, upper bound: 0.0546850
time: 0.23 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0549348, upper bound: 0.0549297
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0550777, upper bound: 0.0551111
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0552443, upper bound: 0.0554759
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0552773, upper bound: 0.0549327
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0546440, upper bound: 0.0553490
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0552454, upper bound: 0.0554578
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0552222
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0552339, upper bound: 0.0552118
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0549839, upper bound: 0.0554704
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0549760, upper bound: 0.0550693
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0549864, upper bound: 0.0554409
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0549864, upper bound: 0.0553811
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0552469, upper bound: 0.0553543
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0552468, upper bound: 0.0547019
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0545979, upper bound: 0.0552446
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0551824, upper bound: 0.0548004
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0550562, upper bound: 0.0551621
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0552118, upper bound: 0.0552339
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0552315, upper bound: 0.0552797
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0551623, upper bound: 0.0552787
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0556942, upper bound: 0.0553809
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0556929, upper bound: 0.0553985
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0554200, upper bound: 0.0547311
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0554407, upper bound: 0.0553030
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0558089, upper bound: 0.0552954
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0557230, upper bound: 0.0553423
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0556211, upper bound: 0.0554509
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0548246, upper bound: 0.0547333
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0550321, upper bound: 0.0542705
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0550036, upper bound: 0.0542718
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0555004, upper bound: 0.0553922
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.32
Output dim: 0, lower bound: -0.0551182, upper bound: 0.0546850

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551964, upper bound: 0.0554575
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552213, upper bound: 0.0554305
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552372, upper bound: 0.0546269
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552619, upper bound: 0.0549209
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545979, upper bound: 0.0552956
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546420, upper bound: 0.0553485
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548125, upper bound: 0.0551597
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542334, upper bound: 0.0551539
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0552124
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0552211
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546344, upper bound: 0.0552081
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552339, upper bound: 0.0551208
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548288, upper bound: 0.0542906
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553438
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0551202
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553143
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0552259
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547601, upper bound: 0.0552486
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546019, upper bound: 0.0549518
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552288, upper bound: 0.0552601
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548355, upper bound: 0.0542546
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548423, upper bound: 0.0543034
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545979, upper bound: 0.0552161
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545979, upper bound: 0.0552436
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551824, upper bound: 0.0547942
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545979, upper bound: 0.0545979
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0545770
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0551621
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551208, upper bound: 0.0552339
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552081, upper bound: 0.0546344
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546975, upper bound: 0.0552194
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552048, upper bound: 0.0552541
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547547, upper bound: 0.0552194
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551353, upper bound: 0.0552531
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555609, upper bound: 0.0551818
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554400, upper bound: 0.0552371
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546696, upper bound: 0.0549641
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554680, upper bound: 0.0549756
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554200, upper bound: 0.0547298
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554137, upper bound: 0.0547298
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554382, upper bound: 0.0552782
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554119, upper bound: 0.0547298
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547107, upper bound: 0.0545926
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555440, upper bound: 0.0551757
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555780, upper bound: 0.0553189
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548158, upper bound: 0.0553077
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555021, upper bound: 0.0548044
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554949, upper bound: 0.0553316
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552539, upper bound: 0.0552666
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553905, upper bound: 0.0551700
time: 0.24 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0551964, upper bound: 0.0554575
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0552213, upper bound: 0.0554305
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0552372, upper bound: 0.0546269
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0552619, upper bound: 0.0549209
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0545979, upper bound: 0.0552956
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0546420, upper bound: 0.0553485
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0548125, upper bound: 0.0551597
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0542334, upper bound: 0.0551539
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0552124
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0552211
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0546344, upper bound: 0.0552081
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0552339, upper bound: 0.0551208
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0548288, upper bound: 0.0542906
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553438
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0551202
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553143
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0552259
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0547601, upper bound: 0.0552486
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0546019, upper bound: 0.0549518
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0552288, upper bound: 0.0552601
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0548355, upper bound: 0.0542546
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0548423, upper bound: 0.0543034
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0545979, upper bound: 0.0552161
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0545979, upper bound: 0.0552436
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0551824, upper bound: 0.0547942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0545979, upper bound: 0.0545979
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0545770
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0551621
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0551208, upper bound: 0.0552339
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0552081, upper bound: 0.0546344
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0546975, upper bound: 0.0552194
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0552048, upper bound: 0.0552541
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0547547, upper bound: 0.0552194
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0551353, upper bound: 0.0552531
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0555609, upper bound: 0.0551818
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0554400, upper bound: 0.0552371
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0546696, upper bound: 0.0549641
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0554680, upper bound: 0.0549756
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0554200, upper bound: 0.0547298
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0554137, upper bound: 0.0547298
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0554382, upper bound: 0.0552782
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0554119, upper bound: 0.0547298
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0547107, upper bound: 0.0545926
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0555440, upper bound: 0.0551757
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0555780, upper bound: 0.0553189
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0548158, upper bound: 0.0553077
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0555021, upper bound: 0.0548044
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0554949, upper bound: 0.0553316
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0552539, upper bound: 0.0552666
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.48
Output dim: 0, lower bound: -0.0553905, upper bound: 0.0551700

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544537, upper bound: 0.0543050
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551214
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551041
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551221
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548183, upper bound: 0.0542136
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548418, upper bound: 0.0544759
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548478, upper bound: 0.0542870
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0549015
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0548542
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546132, upper bound: 0.0553239
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546189, upper bound: 0.0552845
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548125, upper bound: 0.0550837
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548008
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548008
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547961
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548006
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546328, upper bound: 0.0552071
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546253, upper bound: 0.0552079
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552312, upper bound: 0.0551064
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548057, upper bound: 0.0549506
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0549125
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553438
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542370, upper bound: 0.0549313
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553143
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0551930
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551102
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552184
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552015, upper bound: 0.0552365
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545871, upper bound: 0.0546945
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0551967
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0550517
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552194
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545926, upper bound: 0.0547066
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551549, upper bound: 0.0547719
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544739, upper bound: 0.0541406
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546670, upper bound: 0.0547216
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549506, upper bound: 0.0548057
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551064, upper bound: 0.0552312
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552079, upper bound: 0.0546253
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552071, upper bound: 0.0546328
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542829, upper bound: 0.0548026
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548006
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547923, upper bound: 0.0549428
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547570, upper bound: 0.0548418
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546773, upper bound: 0.0552145
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547519, upper bound: 0.0545760
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547427, upper bound: 0.0547886
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547300, upper bound: 0.0548284
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551599, upper bound: 0.0541663
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553095, upper bound: 0.0547809
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551422, upper bound: 0.0548168
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551288, upper bound: 0.0548168
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554399, upper bound: 0.0549499
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551952, upper bound: 0.0549517
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552770, upper bound: 0.0545721
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547935, upper bound: 0.0545721
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547154, upper bound: 0.0543139
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550225, upper bound: 0.0543139
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552845, upper bound: 0.0546189
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552905, upper bound: 0.0551279
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549612, upper bound: 0.0543139
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550209, upper bound: 0.0543211
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0547237
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552914, upper bound: 0.0547648
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552684, upper bound: 0.0548833
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548922, upper bound: 0.0548652
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546656, upper bound: 0.0551828
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547370, upper bound: 0.0551971
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554195, upper bound: 0.0547234
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546269, upper bound: 0.0552205
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554182, upper bound: 0.0551848
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548157, upper bound: 0.0549955
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0541663
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550139, upper bound: 0.0547865
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541687, upper bound: 0.0542634
time: 0.24 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0544537, upper bound: 0.0543050
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551214
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551041
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551221
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0548183, upper bound: 0.0542136
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0548418, upper bound: 0.0544759
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0548478, upper bound: 0.0542870
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0549015
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0548542
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0546132, upper bound: 0.0553239
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0546189, upper bound: 0.0552845
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0548125, upper bound: 0.0550837
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548008
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548008
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547961
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548006
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0546328, upper bound: 0.0552071
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0546253, upper bound: 0.0552079
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0552312, upper bound: 0.0551064
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0548057, upper bound: 0.0549506
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0549125
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553438
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0542370, upper bound: 0.0549313
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553143
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0551930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551102
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552184
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0552015, upper bound: 0.0552365
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0545871, upper bound: 0.0546945
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0551967
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0550517
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552194
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0545926, upper bound: 0.0547066
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0551549, upper bound: 0.0547719
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0544739, upper bound: 0.0541406
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0546670, upper bound: 0.0547216
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0549506, upper bound: 0.0548057
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0551064, upper bound: 0.0552312
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0552079, upper bound: 0.0546253
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0552071, upper bound: 0.0546328
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0542829, upper bound: 0.0548026
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548006
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547923, upper bound: 0.0549428
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547570, upper bound: 0.0548418
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0546773, upper bound: 0.0552145
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547519, upper bound: 0.0545760
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547427, upper bound: 0.0547886
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547300, upper bound: 0.0548284
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0551599, upper bound: 0.0541663
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0553095, upper bound: 0.0547809
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0551422, upper bound: 0.0548168
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0551288, upper bound: 0.0548168
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0554399, upper bound: 0.0549499
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0551952, upper bound: 0.0549517
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0552770, upper bound: 0.0545721
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547935, upper bound: 0.0545721
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547154, upper bound: 0.0543139
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0550225, upper bound: 0.0543139
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0552845, upper bound: 0.0546189
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0552905, upper bound: 0.0551279
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0549612, upper bound: 0.0543139
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0550209, upper bound: 0.0543211
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0547237
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0552914, upper bound: 0.0547648
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0552684, upper bound: 0.0548833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0548922, upper bound: 0.0548652
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0546656, upper bound: 0.0551828
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0547370, upper bound: 0.0551971
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0554195, upper bound: 0.0547234
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0546269, upper bound: 0.0552205
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0554182, upper bound: 0.0551848
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0548157, upper bound: 0.0549955
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0541663
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0550139, upper bound: 0.0547865
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.70
Output dim: 0, lower bound: -0.0541687, upper bound: 0.0542634

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542038, upper bound: 0.0548952
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542000, upper bound: 0.0549096
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542075, upper bound: 0.0548835
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548835
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551295
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542212, upper bound: 0.0548013
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548013
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542155, upper bound: 0.0548013
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548013
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547838, upper bound: 0.0547254
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542871
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0552914
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553133
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0552784
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0552847
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547600, upper bound: 0.0550061
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0551918
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542023, upper bound: 0.0549258
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552158
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548073, upper bound: 0.0547906
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547922
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547974
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547962
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542871, upper bound: 0.0541406
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547254, upper bound: 0.0547838
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0541406
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0542155
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0541406
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0542212
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542681, upper bound: 0.0547883
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542198, upper bound: 0.0547920
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551066, upper bound: 0.0541406
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552847, upper bound: 0.0547539
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551295, upper bound: 0.0547485
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553133, upper bound: 0.0547746
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542548, upper bound: 0.0547926
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550494, upper bound: 0.0547786
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543823, upper bound: 0.0547943
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545854, upper bound: 0.0541406
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548750, upper bound: 0.0541406
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548835, upper bound: 0.0541406
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548835, upper bound: 0.0542075
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548810, upper bound: 0.0547421
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548231, upper bound: 0.0542690
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552158, upper bound: 0.0547237
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548953, upper bound: 0.0547179
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552784, upper bound: 0.0547548
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552914, upper bound: 0.0547648
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542282, upper bound: 0.0548064
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551918, upper bound: 0.0547961
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542561, upper bound: 0.0547632
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0543902
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543283, upper bound: 0.0547789
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541489, upper bound: 0.0547809
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550898, upper bound: 0.0541988
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542136, upper bound: 0.0548183
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551000, upper bound: 0.0547888
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543050, upper bound: 0.0544537
time: 0.27 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.00 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542038, upper bound: 0.0548952
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542000, upper bound: 0.0549096
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542075, upper bound: 0.0548835
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548835
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551295
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542212, upper bound: 0.0548013
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548013
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542155, upper bound: 0.0548013
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548013
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0547838, upper bound: 0.0547254
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542871
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0552914
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553133
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0552784
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0552847
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0547600, upper bound: 0.0550061
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0551918
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542023, upper bound: 0.0549258
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552158
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548073, upper bound: 0.0547906
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547922
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547974
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547962
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542871, upper bound: 0.0541406
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0547254, upper bound: 0.0547838
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0541406
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0542155
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0541406
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0542212
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542681, upper bound: 0.0547883
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542198, upper bound: 0.0547920
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0551066, upper bound: 0.0541406
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0552847, upper bound: 0.0547539
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0551295, upper bound: 0.0547485
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0553133, upper bound: 0.0547746
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542548, upper bound: 0.0547926
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0550494, upper bound: 0.0547786
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0543823, upper bound: 0.0547943
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0545854, upper bound: 0.0541406
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548750, upper bound: 0.0541406
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548835, upper bound: 0.0541406
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548835, upper bound: 0.0542075
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548810, upper bound: 0.0547421
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548231, upper bound: 0.0542690
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0552158, upper bound: 0.0547237
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0548953, upper bound: 0.0547179
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0552784, upper bound: 0.0547548
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0552914, upper bound: 0.0547648
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542282, upper bound: 0.0548064
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0551918, upper bound: 0.0547961
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542561, upper bound: 0.0547632
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0543902
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0543283, upper bound: 0.0547789
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541489, upper bound: 0.0547809
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0550898, upper bound: 0.0541988
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0542136, upper bound: 0.0548183
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0551000, upper bound: 0.0547888
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.00
Output dim: 0, lower bound: -0.0543050, upper bound: 0.0544537

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547437, upper bound: 0.0543289
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546370, upper bound: 0.0552717
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546568, upper bound: 0.0551799
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546437, upper bound: 0.0551868
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546413, upper bound: 0.0551485
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545892, upper bound: 0.0551510
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546745, upper bound: 0.0552754
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546752, upper bound: 0.0547348
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547533, upper bound: 0.0551546
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547588, upper bound: 0.0551496
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546118, upper bound: 0.0550788
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545468, upper bound: 0.0550833
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543703, upper bound: 0.0540591
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543703, upper bound: 0.0540293
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550791, upper bound: 0.0545214
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550523, upper bound: 0.0545287
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550833, upper bound: 0.0545468
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550788, upper bound: 0.0546118
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552691, upper bound: 0.0547548
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544644, upper bound: 0.0546933
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551762, upper bound: 0.0547288
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552560, upper bound: 0.0547258
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548765, upper bound: 0.0547388
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551635, upper bound: 0.0547001
time: 0.25 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.99 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0547437, upper bound: 0.0543289
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0546370, upper bound: 0.0552717
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0546568, upper bound: 0.0551799
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0546437, upper bound: 0.0551868
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0546413, upper bound: 0.0551485
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0545892, upper bound: 0.0551510
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0546745, upper bound: 0.0552754
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0546752, upper bound: 0.0547348
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0547533, upper bound: 0.0551546
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0547588, upper bound: 0.0551496
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0546118, upper bound: 0.0550788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0545468, upper bound: 0.0550833
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0543703, upper bound: 0.0540591
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0543703, upper bound: 0.0540293
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0550791, upper bound: 0.0545214
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0550523, upper bound: 0.0545287
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0550833, upper bound: 0.0545468
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0550788, upper bound: 0.0546118
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0552691, upper bound: 0.0547548
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0544644, upper bound: 0.0546933
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0551762, upper bound: 0.0547288
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0552560, upper bound: 0.0547258
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0548765, upper bound: 0.0547388
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 1.99
Output dim: 0, lower bound: -0.0551635, upper bound: 0.0547001

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545318, upper bound: 0.0552716
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544013, upper bound: 0.0547071
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546354, upper bound: 0.0540278
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545297, upper bound: 0.0551597
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544155, upper bound: 0.0549091
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544071, upper bound: 0.0549525
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540293, upper bound: 0.0543636
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540591, upper bound: 0.0543628
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551396, upper bound: 0.0545892
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551376, upper bound: 0.0546413
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551562, upper bound: 0.0545930
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540831, upper bound: 0.0547076
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552526, upper bound: 0.0547089
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548346, upper bound: 0.0547245
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548454, upper bound: 0.0542674
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548871, upper bound: 0.0543444
time: 0.22 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 1.63 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0545318, upper bound: 0.0552716
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0544013, upper bound: 0.0547071
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0546354, upper bound: 0.0540278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0545297, upper bound: 0.0551597
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0544155, upper bound: 0.0549091
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0544071, upper bound: 0.0549525
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0540293, upper bound: 0.0543636
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0540591, upper bound: 0.0543628
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0551396, upper bound: 0.0545892
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0551376, upper bound: 0.0546413
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0551562, upper bound: 0.0545930
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0540831, upper bound: 0.0547076
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0552526, upper bound: 0.0547089
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0548346, upper bound: 0.0547245
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0548454, upper bound: 0.0542674
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 1.63
Output dim: 0, lower bound: -0.0548871, upper bound: 0.0543444

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543587, upper bound: 0.0548516
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545146, upper bound: 0.0552685
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539362, upper bound: 0.0542452
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539365, upper bound: 0.0542443
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551162, upper bound: 0.0545546
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551257, upper bound: 0.0545980
time: 0.28 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 1.78 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.78
Output dim: 0, lower bound: -0.0543587, upper bound: 0.0548516
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 11, time: 1.78
Output dim: 0, lower bound: -0.0545146, upper bound: 0.0552685
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.78
Output dim: 0, lower bound: -0.0539362, upper bound: 0.0542452
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.78
Output dim: 0, lower bound: -0.0539365, upper bound: 0.0542443
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 1.78
Output dim: 0, lower bound: -0.0551162, upper bound: 0.0545546
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 1.78
Output dim: 0, lower bound: -0.0551257, upper bound: 0.0545980

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 27
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 6
type: DSZ, layer: 3, pos: 39
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 28
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 2
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541738, upper bound: 0.0549388
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541518, upper bound: 0.0549466
time: 0.26 seconds

## Summary of splitting (split count: 11)
- Time for DS candidates: 1.86 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 12, time: 1.86
Output dim: 0, lower bound: -0.0541738, upper bound: 0.0549388
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 12, time: 1.86
Output dim: 0, lower bound: -0.0541518, upper bound: 0.0549466

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.62 + 232.79 = 234.41 seconds
