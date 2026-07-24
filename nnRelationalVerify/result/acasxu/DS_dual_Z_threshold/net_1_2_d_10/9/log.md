## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 57.280903066


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068)
1: (-16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383)
2: (-16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280)
3: (-27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894)
4: (-25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 1.57 = 3.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -57.5687468, upper bound: 57.5687468

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5684507, upper bound: 57.5684509
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5684509, upper bound: 57.5684507
time: 0.46 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.13 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 0, lower bound: -57.5684507, upper bound: 57.5684509
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 0, lower bound: -57.5684509, upper bound: 57.5684507

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.53 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.52 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.49 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.70 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243014
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243015
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243015
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243015
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243015
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.47 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243014
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243015
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243015
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243015
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243015
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.44
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243014
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243014
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243015
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243014
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243014
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4249334
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
time: 0.50 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243014
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243014
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243015
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243014
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4249334, upper bound: 57.4243014
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243014, upper bound: 57.4249334
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243014
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4249334
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -57.4243015, upper bound: 57.4243015

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.87 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.01 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.01
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
time: 0.55 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -57.4037611, upper bound: 57.4037611

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3961807, upper bound: 57.3961807
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068
1: -16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383
2: -16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280
3: -27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894
4: -25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331

Time for backsubstitution: 1.69 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.00 + 417.47 = 420.46 seconds
