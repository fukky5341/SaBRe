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
execution time: IAR + RelationalAnalysis = 0.68 + 0.97 = 1.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.0558836, upper bound: 1.0558836

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550543, upper bound: 1.0558825
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550543, upper bound: 1.0550543
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -1.0550543, upper bound: 1.0558825
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -1.0550543, upper bound: 1.0550543

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538052
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538009
time: 0.35 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0548323, upper bound: 1.0548814
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0548814, upper bound: 1.0548323
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.25 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.25
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538052
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.25
Output dim: 0, lower bound: -1.0538449, upper bound: 1.0538009
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.25
Output dim: 0, lower bound: -1.0548323, upper bound: 1.0548814
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.25
Output dim: 0, lower bound: -1.0548814, upper bound: 1.0548323

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533842, upper bound: 1.0535112
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533842, upper bound: 1.0533797
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480642, upper bound: 1.0479619
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480642, upper bound: 1.0479619
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513144, upper bound: 1.0512876
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513144, upper bound: 1.0512876
time: 0.26 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.09 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -1.0533842, upper bound: 1.0535112
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -1.0533842, upper bound: 1.0533797
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.09
Output dim: 0, lower bound: -1.0480642, upper bound: 1.0479619
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.09
Output dim: 0, lower bound: -1.0480642, upper bound: 1.0479619
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 1.09
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 1.09
Output dim: 0, lower bound: -1.0380338, upper bound: 1.0380338
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -1.0513144, upper bound: 1.0512876
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -1.0513144, upper bound: 1.0512876

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0532751, upper bound: 1.0533977
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0532349, upper bound: 1.0532941
time: 0.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465481, upper bound: 1.0462097
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0466554, upper bound: 1.0462097
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474200, upper bound: 1.0474103
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474200, upper bound: 1.0474103
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0474200
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0474200
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.23 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.23
Output dim: 0, lower bound: -1.0532751, upper bound: 1.0533977
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.23
Output dim: 0, lower bound: -1.0532349, upper bound: 1.0532941
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.23
Output dim: 0, lower bound: -1.0465481, upper bound: 1.0462097
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.23
Output dim: 0, lower bound: -1.0466554, upper bound: 1.0462097
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.23
Output dim: 0, lower bound: -1.0474200, upper bound: 1.0474103
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.23
Output dim: 0, lower bound: -1.0474200, upper bound: 1.0474103
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.23
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0474200
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.23
Output dim: 0, lower bound: -1.0474103, upper bound: 1.0474200

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0460829, upper bound: 1.0469103
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0460829, upper bound: 1.0469103
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489
1: -0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965
2: -0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164
3: -0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897
4: -0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0460829, upper bound: 1.0460829
time: 0.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0460829, upper bound: 1.0460829
time: 0.24 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.41
Output dim: 0, lower bound: -1.0460829, upper bound: 1.0469103
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.41
Output dim: 0, lower bound: -1.0460829, upper bound: 1.0469103
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.41
Output dim: 0, lower bound: -1.0460829, upper bound: 1.0460829
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.41
Output dim: 0, lower bound: -1.0460829, upper bound: 1.0460829

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.65 + 16.13 = 17.78 seconds
