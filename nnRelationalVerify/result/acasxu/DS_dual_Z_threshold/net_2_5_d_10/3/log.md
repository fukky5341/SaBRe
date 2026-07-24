## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.917404983


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732)
1: (-0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920)
2: (-0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294)
3: (-0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840)
4: (-0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.04 + 0.93 = 2.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9266717, upper bound: 0.9266717

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9260939, upper bound: 0.9261088
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9261088, upper bound: 0.9260939
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.75 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.9260939, upper bound: 0.9261088
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.9261088, upper bound: 0.9260939

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9260257, upper bound: 0.9261088
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9260257, upper bound: 0.9257705
time: 0.29 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257705, upper bound: 0.9260939
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9261088, upper bound: 0.9260257
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.66 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.9260257, upper bound: 0.9261088
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.9260257, upper bound: 0.9257705
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.9257705, upper bound: 0.9260939
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.9261088, upper bound: 0.9260257

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255869, upper bound: 0.9257488
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9256782, upper bound: 0.9255089
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255089, upper bound: 0.9254006
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255089, upper bound: 0.9254226
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254226, upper bound: 0.9257397
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254006, upper bound: 0.9255478
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255089, upper bound: 0.9256782
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255089, upper bound: 0.9255869
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.65 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.9255869, upper bound: 0.9257488
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.9256782, upper bound: 0.9255089
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.9255089, upper bound: 0.9254006
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.9255089, upper bound: 0.9254226
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.9254226, upper bound: 0.9257397
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.9254006, upper bound: 0.9255478
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.9255089, upper bound: 0.9256782
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.9255089, upper bound: 0.9255869

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9257277
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255643, upper bound: 0.9255009
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9254326
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255643, upper bound: 0.9251229
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9251229
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254963, upper bound: 0.9251229
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9252430
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9251229
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9257211
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9252430, upper bound: 0.9254988
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9254963
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9251229
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9256488
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254326, upper bound: 0.9254336
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255009, upper bound: 0.9255643
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254326, upper bound: 0.9251229
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9257277
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9255643, upper bound: 0.9255009
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9254326
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9255643, upper bound: 0.9251229
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9251229
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9254963, upper bound: 0.9251229
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9252430
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9251229
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9257211
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9252430, upper bound: 0.9254988
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9254963
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9251229
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9256488
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9254326, upper bound: 0.9254336
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9255009, upper bound: 0.9255643
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 0, lower bound: -0.9254326, upper bound: 0.9251229

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9256093
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250686, upper bound: 0.9252014
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254361, upper bound: 0.9248901
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250766, upper bound: 0.9252758
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250831, upper bound: 0.9248901
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9253003, upper bound: 0.9248901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255251, upper bound: 0.9248901
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254050, upper bound: 0.9248901
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250669, upper bound: 0.9250354
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251810, upper bound: 0.9248901
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9256009, upper bound: 0.9248901
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9256009
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9251810
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250354, upper bound: 0.9250669
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254050
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9255251
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253003
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9250831
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9252758, upper bound: 0.9250766
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254361
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9252014, upper bound: 0.9250686
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9256093, upper bound: 0.9248901
time: 0.28 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9256093
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9250686, upper bound: 0.9252014
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9254361, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9250766, upper bound: 0.9252758
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9250831, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9253003, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9255251, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9254050, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9250669, upper bound: 0.9250354
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9251810, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9256009, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9256009
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9251810
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9250354, upper bound: 0.9250669
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254050
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9255251
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253003
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9250831
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9252758, upper bound: 0.9250766
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254361
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9252014, upper bound: 0.9250686
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 0, lower bound: -0.9256093, upper bound: 0.9248901

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254494
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250686, upper bound: 0.9252014
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9253858, upper bound: 0.9248901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9253192, upper bound: 0.9248901
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250766, upper bound: 0.9252758
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250831, upper bound: 0.9248901
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9253003, upper bound: 0.9248901
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254528, upper bound: 0.9248901
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254368, upper bound: 0.9248901
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9253588, upper bound: 0.9248901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250669, upper bound: 0.9250354
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251810, upper bound: 0.9248901
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251322, upper bound: 0.9248901
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255043, upper bound: 0.9248901
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254299, upper bound: 0.9248901
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254299
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9255043
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9251322
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9251810
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250354, upper bound: 0.9250669
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.40 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253588
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.43 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.53 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.50 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 3.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.52 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254368
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254528
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253003
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9250831
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9252758, upper bound: 0.9250766
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253192
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253858
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9250686
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254494, upper bound: 0.9248901
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254494
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9250686, upper bound: 0.9252014
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9253858, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9253192, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9250766, upper bound: 0.9252758
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9250831, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9253003, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9254528, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9254368, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9253588, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9250669, upper bound: 0.9250354
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9251810, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9251322, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9255043, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9254299, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254299
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9255043
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9251322
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9251810
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9250354, upper bound: 0.9250669
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253588
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254368
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254528
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253003
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9250831
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9252758, upper bound: 0.9250766
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253192
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253858
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9250686
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9254494, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9243601, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9246656, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9246656, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244284, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244284, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9243851, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9242435, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9246067, upper bound: 0.9241919
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9242361, upper bound: 0.9241919
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242435
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9243851
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246274
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246274
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246656
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246656
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9243601
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.32 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9243601, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9246656, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9246656, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9244284, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9244284, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9243851, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9242435, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9246067, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9242361, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242435
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9243851
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246274
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246274
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246656
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246656
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9243601
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9147775
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9147589, upper bound: 0.9146190
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9147172, upper bound: 0.9148370
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9148188
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9147591, upper bound: 0.9146190
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146493
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9147778
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9147619, upper bound: 0.9147864
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9147969
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9148884, upper bound: 0.9146190
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9148655, upper bound: 0.9146190
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9148470, upper bound: 0.9146190
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146972, upper bound: 0.9147217
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 2.15 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.97 + 417.48 = 420.45 seconds
