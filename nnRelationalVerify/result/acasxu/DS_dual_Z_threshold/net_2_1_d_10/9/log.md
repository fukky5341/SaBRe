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
execution time: IAR + RelationalAnalysis = 1.26 + 2.09 = 3.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -146.7073772, upper bound: 146.7073772

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7072806, upper bound: 146.7073184
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7072806, upper bound: 146.7072806
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.38 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 0, lower bound: -146.7072806, upper bound: 146.7073184
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 0, lower bound: -146.7072806, upper bound: 146.7072806

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7070150, upper bound: 146.7070222
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7070150, upper bound: 146.7070222
time: 0.67 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7070222, upper bound: 146.7070163
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.7070222, upper bound: 146.7070150
time: 1.00 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.00 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -146.7070150, upper bound: 146.7070222
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -146.7070150, upper bound: 146.7070222
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -146.7070222, upper bound: 146.7070163
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -146.7070222, upper bound: 146.7070150

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6984392
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6984392
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6983418
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6983418
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6983418
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6983418
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6984392, upper bound: 146.6983418
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6984392, upper bound: 146.6983418
time: 0.89 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.04 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6984392
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6984392
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6983418
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6983418
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6983418
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -146.6983418, upper bound: 146.6983418
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -146.6984392, upper bound: 146.6983418
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -146.6984392, upper bound: 146.6983418

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6954482
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6954482
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6954482, upper bound: 146.6953017
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6954482, upper bound: 146.6953017
time: 0.92 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.17 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6954482
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6954482
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6954482, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -146.6954482, upper bound: 146.6953017

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6954482
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6954482
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6954482, upper bound: 146.6953017
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6954482, upper bound: 146.6953017
time: 0.66 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6954482
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6954482
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6954482, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6953017, upper bound: 146.6953017
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -146.6954482, upper bound: 146.6953017

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951613
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6952961
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951613
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6952961
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6952961, upper bound: 146.6951575
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
time: 1.03 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951613
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6952961
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951613
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6952961
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6952961, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -146.6951575, upper bound: 146.6951575

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
time: 0.99 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.43
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -71.6695175, 94.7768250, -71.6695175, 94.7768250, -166.4463196, 166.4463348
1: -57.0642357, 78.3037186, -57.0642357, 78.3037186, -135.3679504, 135.3679504
2: -47.0594788, 78.8825378, -47.0594788, 78.8825378, -125.9420090, 125.9420090
3: -74.3634262, 94.6706924, -74.3634262, 94.6706924, -169.0341187, 169.0341187
4: -62.3425293, 105.4673996, -62.3425293, 105.4673996, -167.8099365, 167.8099365

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
time: 0.90 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.17 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.17
Output dim: 0, lower bound: -146.6160252, upper bound: 146.6160252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.17
Output dim: 0, lower bound: -146.6177558, upper bound: 146.6177558

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.35 + 416.79 = 420.14 seconds
