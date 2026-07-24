## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 24053.216940845126


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844)
1: (-1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172)
2: (-2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422)
3: (-2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164)
4: (-2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.72 + 2.20 = 4.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -24054.9007839, upper bound: 24054.9007839

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8960125, upper bound: 24054.8958668
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8958668, upper bound: 24054.8960125
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.85 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 0, lower bound: -24054.8960125, upper bound: 24054.8958668
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 0, lower bound: -24054.8958668, upper bound: 24054.8960125

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8808626, upper bound: 24054.8791033
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8792901, upper bound: 24054.8791231
time: 0.89 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8791231, upper bound: 24054.8792901
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8791033, upper bound: 24054.8808626
time: 0.80 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.51 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.51
Output dim: 0, lower bound: -24054.8808626, upper bound: 24054.8791033
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.51
Output dim: 0, lower bound: -24054.8792901, upper bound: 24054.8791231
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.51
Output dim: 0, lower bound: -24054.8791231, upper bound: 24054.8792901
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.51
Output dim: 0, lower bound: -24054.8791033, upper bound: 24054.8808626

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8733957, upper bound: 24054.8730820
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8733957, upper bound: 24054.8730820
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8730820
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8730820
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8730820
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8730820
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8733956
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8733956
time: 0.80 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.54 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -24054.8733957, upper bound: 24054.8730820
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -24054.8733957, upper bound: 24054.8730820
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8730820
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8730820
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8730820
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8730820
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8733956
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.54
Output dim: 0, lower bound: -24054.8730820, upper bound: 24054.8733956

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8543241, upper bound: 24054.8542637
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8545020, upper bound: 24054.8540844
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8543547
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8543547, upper bound: 24054.8540844
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8545020
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8543420
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8542637, upper bound: 24054.8543241
time: 0.73 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.26 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8543241, upper bound: 24054.8542637
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8545020, upper bound: 24054.8540844
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8543547
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8543547, upper bound: 24054.8540844
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8545020
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8543420
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8540844, upper bound: 24054.8540844
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -24054.8542637, upper bound: 24054.8543241

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8542493, upper bound: 24054.8541827
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540242, upper bound: 24054.8540100
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8542668, upper bound: 24054.8540100
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8544240, upper bound: 24054.8540100
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8543023, upper bound: 24054.8540100
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8542764
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8542764, upper bound: 24054.8540100
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8543023
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8544240
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8542668
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540242
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8541827, upper bound: 24054.8542493
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
time: 0.88 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8542493, upper bound: 24054.8541827
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540242, upper bound: 24054.8540100
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8542668, upper bound: 24054.8540100
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8544240, upper bound: 24054.8540100
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8543023, upper bound: 24054.8540100
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8542764
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8542764, upper bound: 24054.8540100
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8543023
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8544240
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8542668
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540242
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8541827, upper bound: 24054.8542493
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.99
Output dim: 0, lower bound: -24054.8540100, upper bound: 24054.8540100

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8507812, upper bound: 24054.8506013
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8508192, upper bound: 24054.8506013
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8507414, upper bound: 24054.8506013
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506145, upper bound: 24054.8506013
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8507019, upper bound: 24054.8506013
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506792, upper bound: 24054.8506013
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506792
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8507019
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506145
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8507414
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8508192
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8507812
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.90 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8507812, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8508192, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8507414, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506145, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8507019, upper bound: 24054.8506013
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506792, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506792
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8507019
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506145
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8507414
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8508192
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8507812
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8507812, upper bound: 24054.8506013
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8508192, upper bound: 24054.8506013
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8507414, upper bound: 24054.8506013
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506145, upper bound: 24054.8506013
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506115, upper bound: 24054.8506013
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8507019, upper bound: 24054.8506013
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506094, upper bound: 24054.8506013
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506792, upper bound: 24054.8506013
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 15

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8506013, upper bound: 24054.8506013
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844
1: -1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172
2: -2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422
3: -2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164
4: -2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750

Time for backsubstitution: 2.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.23 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.92 + 415.30 = 420.22 seconds
