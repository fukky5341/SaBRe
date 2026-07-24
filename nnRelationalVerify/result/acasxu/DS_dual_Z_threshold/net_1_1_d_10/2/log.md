## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 1.0495482984


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489)
1: (-0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965)
2: (-0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164)
3: (-0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897)
4: (-0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.68 + 0.94 = 1.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.0558836, upper bound: 1.0558836

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.61 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.26 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
time: 0.25 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.12 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507156
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0507328

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0506509
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
time: 0.24 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.14 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -1.0507156, upper bound: 1.0506509
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507079
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -1.0506509, upper bound: 1.0507156
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0506509
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -1.0507079, upper bound: 1.0507328

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0505224
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0505830
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506150
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506197, upper bound: 1.0505224
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0505830
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.26 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0506223, upper bound: 1.0506150
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506150
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0505224
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0505830
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0505582, upper bound: 1.0506150
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0505830, upper bound: 1.0506150
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506162
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0505224, upper bound: 1.0506197
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0506197, upper bound: 1.0505224
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0506162, upper bound: 1.0505582
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0505830
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.26
Output dim: 0, lower bound: -1.0506150, upper bound: 1.0506223

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0504719
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504932
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505234
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504357
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504578
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504607
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505232
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505235, upper bound: 1.0504719
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504932
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505189
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505191
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505232
time: 0.29 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0504719
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504932
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504607, upper bound: 1.0505189
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505234
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505191
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504357
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504578
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504876, upper bound: 1.0504607
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505232
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0505232, upper bound: 1.0505185
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0505235, upper bound: 1.0504719
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0505164
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0494047, upper bound: 1.0504932
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505189
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504578, upper bound: 1.0504656
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505234
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0500394, upper bound: 1.0505191
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0505191, upper bound: 1.0504357
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0505234, upper bound: 1.0500394
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504656, upper bound: 1.0504578
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0505189, upper bound: 1.0504607
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504931, upper bound: 1.0504876
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0505164, upper bound: 1.0494047
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504719, upper bound: 1.0505235
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.58
Output dim: 0, lower bound: -1.0504357, upper bound: 1.0505232

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0479717
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480332
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480399
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479678
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
time: 0.25 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479717, upper bound: 1.0480332
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479710, upper bound: 1.0480399
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0469980
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0474069, upper bound: 1.0476019
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0478006, upper bound: 1.0479678
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0479717
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0474510, upper bound: 1.0469980
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480951, upper bound: 1.0480451
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0479862
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480804, upper bound: 1.0474589
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480485
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0480332
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0474510
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480372
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479558, upper bound: 1.0480399
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479678, upper bound: 1.0478006
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479679, upper bound: 1.0469980
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0475853, upper bound: 1.0480752
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0475947, upper bound: 1.0480752
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0478594, upper bound: 1.0480712
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0476019, upper bound: 1.0474069
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0476019
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480712, upper bound: 1.0478594
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475947
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480752, upper bound: 1.0475853
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479679
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0479678
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480399, upper bound: 1.0479710
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480372, upper bound: 1.0479558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0469980, upper bound: 1.0469980
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480332, upper bound: 1.0479717
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480485, upper bound: 1.0469980
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0474589, upper bound: 1.0480804
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0479862, upper bound: 1.0480804
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.34
Output dim: 0, lower bound: -1.0480451, upper bound: 1.0480951

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.63 + 81.42 = 83.05 seconds
