## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 146.59001129824


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348)
1: (-57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504)
2: (-47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090)
3: (-74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187)
4: (-62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.05 + 2.06 = 3.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -146.7073772, upper bound: 146.7073772

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7064547, upper bound: 146.7064547
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7064547, upper bound: 146.7064552
time: 0.66 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.51 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -146.7064547, upper bound: 146.7064547
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -146.7064547, upper bound: 146.7064552

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7064092, upper bound: 146.7064092
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7064092, upper bound: 146.7064154
time: 0.80 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7055959, upper bound: 146.7055977
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7055977, upper bound: 146.7055973
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.83 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 0, lower bound: -146.7064092, upper bound: 146.7064092
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 0, lower bound: -146.7064092, upper bound: 146.7064154
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 0, lower bound: -146.7055959, upper bound: 146.7055977
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 0, lower bound: -146.7055977, upper bound: 146.7055973

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7062871, upper bound: 146.7062871
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7062871, upper bound: 146.7062871
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7063510, upper bound: 146.7063546
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7063565, upper bound: 146.7063521
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7055906, upper bound: 146.7055977
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7055959, upper bound: 146.7055967
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052395, upper bound: 146.7052452
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052395, upper bound: 146.7052450
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.30 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -146.7062871, upper bound: 146.7062871
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -146.7062871, upper bound: 146.7062871
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -146.7063510, upper bound: 146.7063546
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -146.7063565, upper bound: 146.7063521
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -146.7055906, upper bound: 146.7055977
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -146.7055959, upper bound: 146.7055967
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -146.7052395, upper bound: 146.7052452
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -146.7052395, upper bound: 146.7052450

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050496, upper bound: 146.7050815
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050506, upper bound: 146.7050495
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050721, upper bound: 146.7050802
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050731, upper bound: 146.7050544
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7063310, upper bound: 146.7063321
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7063310, upper bound: 146.7063329
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061386, upper bound: 146.7061394
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061386, upper bound: 146.7061386
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6940401, upper bound: 146.6940401
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6940401, upper bound: 146.6940401
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7055906, upper bound: 146.7055906
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7055959, upper bound: 146.7055967
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052645, upper bound: 146.7052395
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052395, upper bound: 146.7052452
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052395, upper bound: 146.7052450
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052395, upper bound: 146.7052441
time: 0.92 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7050496, upper bound: 146.7050815
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7050506, upper bound: 146.7050495
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7050721, upper bound: 146.7050802
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7050731, upper bound: 146.7050544
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7063310, upper bound: 146.7063321
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7063310, upper bound: 146.7063329
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7061386, upper bound: 146.7061394
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7061386, upper bound: 146.7061386
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.6940401, upper bound: 146.6940401
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.6940401, upper bound: 146.6940401
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7055906, upper bound: 146.7055906
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7055959, upper bound: 146.7055967
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7052645, upper bound: 146.7052395
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7052395, upper bound: 146.7052452
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7052395, upper bound: 146.7052450
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.93
Output dim: 0, lower bound: -146.7052395, upper bound: 146.7052441

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050495, upper bound: 146.7050815
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050708, upper bound: 146.7050804
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050721, upper bound: 146.7050802
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050718, upper bound: 146.7050495
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061240, upper bound: 146.7061240
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061240, upper bound: 146.7061240
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061386, upper bound: 146.7061386
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061386, upper bound: 146.7061394
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061298, upper bound: 146.7061298
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061298, upper bound: 146.7061298
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6940262, upper bound: 146.6940262
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6940262, upper bound: 146.6940262
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050757, upper bound: 146.7050810
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050757, upper bound: 146.7050870
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7055060, upper bound: 146.7055060
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7055060, upper bound: 146.7055143
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7051445, upper bound: 146.7051445
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7051445, upper bound: 146.7051445
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052390, upper bound: 146.7052433
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7052390, upper bound: 146.7052427
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046552
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046529
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
time: 0.82 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7050495, upper bound: 146.7050815
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7050708, upper bound: 146.7050804
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7050721, upper bound: 146.7050802
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7050718, upper bound: 146.7050495
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7061240, upper bound: 146.7061240
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7061240, upper bound: 146.7061240
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7061386, upper bound: 146.7061386
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7061386, upper bound: 146.7061394
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7061298, upper bound: 146.7061298
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7061298, upper bound: 146.7061298
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6940262, upper bound: 146.6940262
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6940262, upper bound: 146.6940262
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7050757, upper bound: 146.7050810
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7050757, upper bound: 146.7050870
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7055060, upper bound: 146.7055060
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7055060, upper bound: 146.7055143
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7051445, upper bound: 146.7051445
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7051445, upper bound: 146.7051445
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7052390, upper bound: 146.7052433
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7052390, upper bound: 146.7052427
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046552
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046529
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050497, upper bound: 146.7050815
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050495, upper bound: 146.7050524
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045870
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050495, upper bound: 146.7050495
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050495, upper bound: 146.7050495
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061240, upper bound: 146.7061240
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061240, upper bound: 146.7061240
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6950649, upper bound: 146.6950649
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6950649, upper bound: 146.6950649
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958307
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958419
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6952072, upper bound: 146.6952072
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6952072, upper bound: 146.6952072
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6156631, upper bound: 146.6156631
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6156631, upper bound: 146.6156631
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061298, upper bound: 146.7061298
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061298, upper bound: 146.7061298
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930231
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050757, upper bound: 146.7050757
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050757, upper bound: 146.7050870
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054860, upper bound: 146.7054860
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054860, upper bound: 146.7054860
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046552
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046552
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046529
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046519
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
time: 0.62 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7050497, upper bound: 146.7050815
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7050495, upper bound: 146.7050524
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045870
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7050495, upper bound: 146.7050495
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7050495, upper bound: 146.7050495
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7061240, upper bound: 146.7061240
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7061240, upper bound: 146.7061240
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6950649, upper bound: 146.6950649
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6950649, upper bound: 146.6950649
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958307
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958419
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6952072, upper bound: 146.6952072
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6952072, upper bound: 146.6952072
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6156631, upper bound: 146.6156631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6156631, upper bound: 146.6156631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7061298, upper bound: 146.7061298
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7061298, upper bound: 146.7061298
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930231
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7050757, upper bound: 146.7050757
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7050757, upper bound: 146.7050870
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7054860, upper bound: 146.7054860
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7054860, upper bound: 146.7054860
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046552
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046552
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046529
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046519
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6931547
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6931547
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928509
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928509
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7049889, upper bound: 146.7049600
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7049600, upper bound: 146.7049600
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958419
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958307
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958307
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6950649, upper bound: 146.6951511
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6950649, upper bound: 146.6950649
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177480, upper bound: 146.6177480
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177480, upper bound: 146.6177480
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6952072, upper bound: 146.6952072
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6952072, upper bound: 146.6952072
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6945370, upper bound: 146.6945370
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6945370, upper bound: 146.6945370
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6156327, upper bound: 146.6156327
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6156327, upper bound: 146.6156327
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7045142, upper bound: 146.7045145
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7045142, upper bound: 146.7045142
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930231
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050343, upper bound: 146.7050343
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7050343, upper bound: 146.7050438
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054860, upper bound: 146.7054860
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7054860, upper bound: 146.7054860
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046552
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046530
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046348, upper bound: 146.7046348
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046348, upper bound: 146.7046506
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046468
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046529
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
time: 0.57 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6931547
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6931547
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930457, upper bound: 146.6930457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928509
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928509
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7049889, upper bound: 146.7049600
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7049600, upper bound: 146.7049600
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958419
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958307
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958307
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6950649, upper bound: 146.6951511
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6950649, upper bound: 146.6950649
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6177480, upper bound: 146.6177480
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6177480, upper bound: 146.6177480
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6952072, upper bound: 146.6952072
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6952072, upper bound: 146.6952072
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6945370, upper bound: 146.6945370
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6945370, upper bound: 146.6945370
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6156327, upper bound: 146.6156327
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6156327, upper bound: 146.6156327
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7045142, upper bound: 146.7045145
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7045142, upper bound: 146.7045142
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930231
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7050343, upper bound: 146.7050343
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7050343, upper bound: 146.7050438
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7054860, upper bound: 146.7054860
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7054860, upper bound: 146.7054860
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046552
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046530
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7046348, upper bound: 146.7046348
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7046348, upper bound: 146.7046506
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046468
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046529
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.50
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928970
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928970
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930365, upper bound: 146.6930365
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930365, upper bound: 146.6930365
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927981
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928502
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927981
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928502
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6943928, upper bound: 146.6943928
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6943928, upper bound: 146.6943928
time: 0.78 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928970
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928970
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6930365, upper bound: 146.6930365
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6930365, upper bound: 146.6930365
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927981
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928502
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927981
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6928502
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138881, upper bound: 146.6138881
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6138981, upper bound: 146.6138981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6930331, upper bound: 146.6930331
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6943928, upper bound: 146.6943928
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.70
Output dim: 0, lower bound: -146.6943928, upper bound: 146.6943928
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6155870, upper bound: 146.6155870
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958984
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958419
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958307
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6958307, upper bound: 146.6958307
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6950649, upper bound: 146.6951511
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6950649, upper bound: 146.6950649
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6177480, upper bound: 146.6177480
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6177480, upper bound: 146.6177480
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6952072, upper bound: 146.6952072
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6952072, upper bound: 146.6952072
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6945370, upper bound: 146.6945370
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6945370, upper bound: 146.6945370
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6139464, upper bound: 146.6139464
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6156327, upper bound: 146.6156327
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6156327, upper bound: 146.6156327
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7061198, upper bound: 146.7061198
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7045142, upper bound: 146.7045145
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7045142, upper bound: 146.7045142
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7045492, upper bound: 146.7045492
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930231
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6930229, upper bound: 146.6930229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7050343, upper bound: 146.7050343
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7050343, upper bound: 146.7050438
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7054860, upper bound: 146.7054860
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7054860, upper bound: 146.7054860
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6927666, upper bound: 146.6927666
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160173, upper bound: 146.6160173
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6934525, upper bound: 146.6934525
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6934613, upper bound: 146.6934613
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6139365, upper bound: 146.6139365
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046552
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046530
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7046348, upper bound: 146.7046348
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7046348, upper bound: 146.7046506
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046468
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.7046468, upper bound: 146.7046529
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6927572, upper bound: 146.6927572
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160613, upper bound: 146.6160613
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -146.6160692, upper bound: 146.6160692

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.11 + 417.16 = 420.27 seconds
