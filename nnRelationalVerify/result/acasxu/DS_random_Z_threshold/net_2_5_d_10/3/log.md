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
execution time: IAR + RelationalAnalysis = 0.78 + 0.80 = 1.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9266717, upper bound: 0.9266717

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9265850, upper bound: 0.9266717
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9265850, upper bound: 0.9265850
time: 0.19 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.40 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.40
Output dim: 0, lower bound: -0.9265850, upper bound: 0.9266717
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.40
Output dim: 0, lower bound: -0.9265850, upper bound: 0.9265850

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9262212, upper bound: 0.9263321
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9262212, upper bound: 0.9262859
time: 0.18 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9262212, upper bound: 0.9262467
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9262212, upper bound: 0.9262212
time: 0.18 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.06 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.06
Output dim: 0, lower bound: -0.9262212, upper bound: 0.9263321
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.06
Output dim: 0, lower bound: -0.9262212, upper bound: 0.9262859
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.06
Output dim: 0, lower bound: -0.9262212, upper bound: 0.9262467
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.06
Output dim: 0, lower bound: -0.9262212, upper bound: 0.9262212

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9258213, upper bound: 0.9262038
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9261035, upper bound: 0.9257508
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9262488
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9257311
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9262174
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9259821
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9262067
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9257311
time: 0.22 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.16 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.16
Output dim: 0, lower bound: -0.9258213, upper bound: 0.9262038
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.16
Output dim: 0, lower bound: -0.9261035, upper bound: 0.9257508
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.16
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9262488
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.16
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9257311
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.16
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9262174
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.16
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9259821
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.16
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9262067
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.16
Output dim: 0, lower bound: -0.9257311, upper bound: 0.9257311

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9261873
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9256656, upper bound: 0.9257100
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9254799
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9260841, upper bound: 0.9255899
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9261165
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9254799
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9258218, upper bound: 0.9254799
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9260894, upper bound: 0.9254799
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9260894
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9258218
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9256036
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9261165, upper bound: 0.9255979
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9260841
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9256656
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9251229
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254326, upper bound: 0.9251229
time: 0.18 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9261873
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9256656, upper bound: 0.9257100
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9254799
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9260841, upper bound: 0.9255899
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9261165
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9254799
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9258218, upper bound: 0.9254799
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9260894, upper bound: 0.9254799
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9260894
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9258218
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9256036
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9261165, upper bound: 0.9255979
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9260841
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9254799, upper bound: 0.9256656
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9251229, upper bound: 0.9251229
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.9254326, upper bound: 0.9251229

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9256093
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9256009
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250686, upper bound: 0.9252014
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9251810
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254361, upper bound: 0.9248901
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250354, upper bound: 0.9250669
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250766, upper bound: 0.9252758
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254050
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250831, upper bound: 0.9248901
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9253003, upper bound: 0.9248901
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255251, upper bound: 0.9248901
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9255251
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253003
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9250831
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254050, upper bound: 0.9248901
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9252758, upper bound: 0.9250766
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250669, upper bound: 0.9250354
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254361
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251810, upper bound: 0.9248901
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9252014, upper bound: 0.9250686
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9256009, upper bound: 0.9248901
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9256093, upper bound: 0.9248901
time: 0.18 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9256093
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9256009
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9250686, upper bound: 0.9252014
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9251810
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9254361, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9250354, upper bound: 0.9250669
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9250766, upper bound: 0.9252758
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254050
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9250831, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9253003, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9255251, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9255251
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253003
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9250831
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9254050, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9252758, upper bound: 0.9250766
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9250669, upper bound: 0.9250354
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254361
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9251810, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9252014, upper bound: 0.9250686
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9256009, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.12
Output dim: 0, lower bound: -0.9256093, upper bound: 0.9248901

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254494
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254299
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9255043
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250686, upper bound: 0.9252014
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9246736, upper bound: 0.9241919
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250354, upper bound: 0.9250669
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253588
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9250831, upper bound: 0.9248901
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9253003, upper bound: 0.9248901
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254528, upper bound: 0.9248901
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254368, upper bound: 0.9248901
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9250831
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9252758, upper bound: 0.9250766
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246736
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246736
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251810, upper bound: 0.9248901
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251322, upper bound: 0.9248901
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9243601
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255043, upper bound: 0.9248901
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9254299, upper bound: 0.9248901
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.19 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254494
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9254299
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9255043
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9250686, upper bound: 0.9252014
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9246736, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9250354, upper bound: 0.9250669
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9253588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9250831, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9253003, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9254528, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9254368, upper bound: 0.9248901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9250831
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9252758, upper bound: 0.9250766
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246736
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246736
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9251810, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9251322, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9243601
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9248901, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9255043, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9254299, upper bound: 0.9248901
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.20
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9243601, upper bound: 0.9241919
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244284, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244284, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9243851, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9242435, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242435
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9243851
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9242136, upper bound: 0.9241919
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246274
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246656
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246274
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246656
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9243601
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
time: 0.20 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9243601, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242136
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9244284, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9244284, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9243851, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9242435, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242435
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246067
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9243851
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9244284
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9242136, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9242361
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246274
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246656
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246274
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9246656
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9243601
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.34
Output dim: 0, lower bound: -0.9241919, upper bound: 0.9241919

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9177207
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9177648
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9177190
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146493
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9147778
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9238666, upper bound: 0.9237077
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241621, upper bound: 0.9237077
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239621, upper bound: 0.9239415
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9178236, upper bound: 0.9176707
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9177993, upper bound: 0.9176707
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176936, upper bound: 0.9176776
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241949
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146376, upper bound: 0.9146190
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9177366, upper bound: 0.9176707
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9177192, upper bound: 0.9176707
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239311, upper bound: 0.9237077
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239310, upper bound: 0.9237077
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9243658, upper bound: 0.9241717
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9243303, upper bound: 0.9241717
time: 0.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9242284, upper bound: 0.9241717
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244886, upper bound: 0.9241717
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239312, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241032, upper bound: 0.9237077
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9241032
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9239312
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146420, upper bound: 0.9146190
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146327
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239611
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9178042
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9178447
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9147630
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239539
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146376
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239449, upper bound: 0.9239415
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239419
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146716, upper bound: 0.9146190
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237086
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9177993
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239645
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9148129
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9148655
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9148884
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239517
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 33

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9148188, upper bound: 0.9146190
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9148370, upper bound: 0.9147172
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 48

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
time: 0.23 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.68 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9177207
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9177648
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9177190
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146493
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9147778
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9238666, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241621, upper bound: 0.9237077
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239621, upper bound: 0.9239415
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9178236, upper bound: 0.9176707
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9177993, upper bound: 0.9176707
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176936, upper bound: 0.9176776
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241949
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146376, upper bound: 0.9146190
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9177366, upper bound: 0.9176707
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9177192, upper bound: 0.9176707
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239311, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239310, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9243658, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9243303, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9242284, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9244886, upper bound: 0.9241717
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239312, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241032, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9241032
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9239312
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146420, upper bound: 0.9146190
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146327
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9178042
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9178447
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9147630
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239539
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146376
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239449, upper bound: 0.9239415
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239419
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146716, upper bound: 0.9146190
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237086
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9177993
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239645
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9148129
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9148655
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9148884
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239517
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9237077, upper bound: 0.9237077
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9146190, upper bound: 0.9146190
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9148188, upper bound: 0.9146190
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9148370, upper bound: 0.9147172
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9241717, upper bound: 0.9241717
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9176707, upper bound: 0.9176707
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 1.68
Output dim: 0, lower bound: -0.9239415, upper bound: 0.9239415

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9165202, upper bound: 0.9165202
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9165202, upper bound: 0.9165202
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9165202, upper bound: 0.9165202
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9165202, upper bound: 0.9165202
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9166222, upper bound: 0.9166222
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9166222, upper bound: 0.9166222
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9236875, upper bound: 0.9236875
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9236875, upper bound: 0.9236875
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8936707, upper bound: 0.8937645
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8936309, upper bound: 0.8937909
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8937371, upper bound: 0.8936309
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8936309, upper bound: 0.8936309
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9236875, upper bound: 0.9236875
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9236875, upper bound: 0.9236875
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239222, upper bound: 0.9239222
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239222, upper bound: 0.9239222
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9232735, upper bound: 0.9232735
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9232735, upper bound: 0.9232735
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9100840, upper bound: 0.9102688
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9100840, upper bound: 0.9103947
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9138183, upper bound: 0.9138183
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9138183, upper bound: 0.9138183
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9165202, upper bound: 0.9165202
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9165202, upper bound: 0.9165892
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9166222, upper bound: 0.9166679
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9166222, upper bound: 0.9166222
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9100840, upper bound: 0.9104246
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9100840, upper bound: 0.9106253
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9105984, upper bound: 0.9103390
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9100840, upper bound: 0.9103443
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9138183, upper bound: 0.9138183
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9138183, upper bound: 0.9138183
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239222, upper bound: 0.9239222
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239222, upper bound: 0.9239222
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9236875, upper bound: 0.9236875
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9236875, upper bound: 0.9236875
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9138183, upper bound: 0.9139548
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9138183, upper bound: 0.9140627
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9165202, upper bound: 0.9165202
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9165202, upper bound: 0.9165202
time: 0.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9232735, upper bound: 0.9232735
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9232735, upper bound: 0.9232735
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9236875, upper bound: 0.9236875
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9236875, upper bound: 0.9236875
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9176513, upper bound: 0.9176513
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9232735, upper bound: 0.9232735
time: 0.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9232735, upper bound: 0.9232735
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9100840, upper bound: 0.9100840
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9100840, upper bound: 0.9100840
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 36

### Candidate
type: DSZ, layer: 5, pos: 2

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9100840, upper bound: 0.9100840
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9100840, upper bound: 0.9100840
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9232735, upper bound: 0.9232735
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9232735, upper bound: 0.9232735
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9138183, upper bound: 0.9138183
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9138183, upper bound: 0.9138183
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 36
type: DSZ, layer: 5, pos: 41
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9138183, upper bound: 0.9138183
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9138183, upper bound: 0.9138183
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 10
type: DSZ, layer: 5, pos: 2
type: DSZ, layer: 5, pos: 18
type: DSZ, layer: 5, pos: 0
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 5, pos: 17

### Candidate
type: DSZ, layer: 5, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9232735, upper bound: 0.9232735
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9232735, upper bound: 0.9232735
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1100792, 1.0819939, -0.1100792, 1.0819939, -1.1920732, 1.1920732
1: -0.1110148, 0.1496772, -0.1110148, 0.1496772, -0.2606920, 0.2606920
2: -0.0314484, 0.2418811, -0.0314484, 0.2418811, -0.2733294, 0.2733294
3: -0.1107760, 0.1208080, -0.1107760, 0.1208080, -0.2315840, 0.2315840
4: -0.0367316, 0.2377871, -0.0367316, 0.2377871, -0.2745188, 0.2745188

Time for backsubstitution: 1.14 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 1.57 + 419.12 = 420.69 seconds
