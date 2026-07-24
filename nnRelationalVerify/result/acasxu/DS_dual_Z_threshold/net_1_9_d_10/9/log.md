## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 2.87951805


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807)
1: (-14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082)
2: (-7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548)
3: (-9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106)
4: (-5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 1.48 = 2.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3.1994645, upper bound: 3.1994645

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1994456, upper bound: 3.1994500
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1994456, upper bound: 3.1994456
time: 0.46 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 0, lower bound: -3.1994456, upper bound: 3.1994500
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 0, lower bound: -3.1994456, upper bound: 3.1994456

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
time: 0.65 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
time: 0.44 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.02 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9388323
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9397741, upper bound: 2.9388323
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9388323
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9397741, upper bound: 2.9388323
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9397741
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9388323
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9397741
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9388323
time: 0.51 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.18 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9388323
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.9397741, upper bound: 2.9388323
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9388323
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.9397741, upper bound: 2.9388323
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9397741
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9388323
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9397741
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -2.9388323, upper bound: 2.9388323

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8911255
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8911255
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8911255, upper bound: 2.8910173
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8911255, upper bound: 2.8910173
time: 0.44 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.09 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8911255
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8911255
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8911255, upper bound: 2.8910173
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -2.8911255, upper bound: 2.8910173

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8911255
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918510, upper bound: 2.8910173
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918304, upper bound: 2.8910173
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8911255
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918510, upper bound: 2.8910173
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918239, upper bound: 2.8910173
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918239
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918510
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8911255, upper bound: 2.8910173
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918304
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918510
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8911255, upper bound: 2.8910173
time: 0.50 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8911255
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8918510, upper bound: 2.8910173
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8918304, upper bound: 2.8910173
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8911255
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8918510, upper bound: 2.8910173
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8918239, upper bound: 2.8910173
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8918545, upper bound: 2.8910173
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918239
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918510
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8911255, upper bound: 2.8910173
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918304
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918545
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8918510
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8910173, upper bound: 2.8910173
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -2.8911255, upper bound: 2.8910173

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839918
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839918
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8835893, upper bound: 2.8839406
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8835916, upper bound: 2.8834496
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8835672, upper bound: 2.8834496
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8839445, upper bound: 2.8834496
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839918
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839918
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8835893, upper bound: 2.8839406
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8835916, upper bound: 2.8834496
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8835599, upper bound: 2.8834496
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8839445, upper bound: 2.8834496
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839445
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8835599
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8835916
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8839406, upper bound: 2.8835893
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8839918, upper bound: 2.8834496
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8839918, upper bound: 2.8834496
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839445
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8835672
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8835916
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8839406, upper bound: 2.8835893
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8839918, upper bound: 2.8834496
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8839918, upper bound: 2.8834496
time: 0.54 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839918
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839918
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8835893, upper bound: 2.8839406
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8835916, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8835672, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8839445, upper bound: 2.8834496
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839918
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839918
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8835893, upper bound: 2.8839406
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8835916, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8835599, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8839445, upper bound: 2.8834496
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839445
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8835599
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8835916
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8839406, upper bound: 2.8835893
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8839918, upper bound: 2.8834496
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8839918, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8839445
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8835672
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8835916
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8839406, upper bound: 2.8835893
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8834496, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8839918, upper bound: 2.8834496
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -2.8839918, upper bound: 2.8834496

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781721
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781839
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8779896, upper bound: 2.8781603
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8781993, upper bound: 2.8776193
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8782014, upper bound: 2.8776193
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8781779, upper bound: 2.8776193
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8782560, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781721
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781839
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8779896, upper bound: 2.8781603
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8781993, upper bound: 2.8776193
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8782014, upper bound: 2.8776193
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8781707, upper bound: 2.8776193
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8782560, upper bound: 2.8776193
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8782560
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781707
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8782014
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781993
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8781603, upper bound: 2.8779896
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8781839, upper bound: 2.8776193
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8781721, upper bound: 2.8776193
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8782560
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781779
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8782014
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781993
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8781603, upper bound: 2.8779896
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8781839, upper bound: 2.8776193
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807
1: -14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082
2: -7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548
3: -9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106
4: -5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8781721, upper bound: 2.8776193
time: 0.63 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.56 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781721
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781839
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8779896, upper bound: 2.8781603
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8781993, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8782014, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8781779, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8782560, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781721
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781839
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8779896, upper bound: 2.8781603
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8781993, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8782014, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8781707, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8782560, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8782560
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781707
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8782014
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781993
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8781603, upper bound: 2.8779896
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8781839, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8781721, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8782560
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781779
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8782014
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8781993
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8781603, upper bound: 2.8779896
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8781839, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8776193, upper bound: 2.8776193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 0, lower bound: -2.8781721, upper bound: 2.8776193

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.64 + 292.60 = 295.24 seconds
