## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 1541.9334605111999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461)
1: (-576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391)
2: (-492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311)
3: (-691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342)
4: (-652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.20 + 2.11 = 3.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1541.9488800, upper bound: 1541.9488800

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9478896, upper bound: 1541.9479269
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9478896, upper bound: 1541.9478896
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.67 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -1541.9478896, upper bound: 1541.9479269
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -1541.9478896, upper bound: 1541.9478896

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335537, upper bound: 1541.9335537
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335537, upper bound: 1541.9335537
time: 0.79 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335537, upper bound: 1541.9335537
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335537, upper bound: 1541.9335537
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.77 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -1541.9335537, upper bound: 1541.9335537
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -1541.9335537, upper bound: 1541.9335537
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -1541.9335537, upper bound: 1541.9335537
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.77
Output dim: 0, lower bound: -1541.9335537, upper bound: 1541.9335537

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334914, upper bound: 1541.9335152
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335152, upper bound: 1541.9334914
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334914, upper bound: 1541.9335152
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335152, upper bound: 1541.9334914
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334914, upper bound: 1541.9335152
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335152, upper bound: 1541.9334914
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334914, upper bound: 1541.9335152
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335152, upper bound: 1541.9334914
time: 0.72 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.69 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1541.9334914, upper bound: 1541.9335152
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1541.9335152, upper bound: 1541.9334914
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1541.9334914, upper bound: 1541.9335152
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1541.9335152, upper bound: 1541.9334914
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1541.9334914, upper bound: 1541.9335152
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1541.9335152, upper bound: 1541.9334914
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1541.9334914, upper bound: 1541.9335152
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -1541.9335152, upper bound: 1541.9334914

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334649, upper bound: 1541.9334354
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.90 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334649, upper bound: 1541.9334354
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.04
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334412
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334412
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334407
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334412, upper bound: 1541.9334354
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334407
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
time: 1.21 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334412
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334412
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334407
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334354
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 0, lower bound: -1541.9334412, upper bound: 1541.9334354
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.81
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334407
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -1541.9334354, upper bound: 1541.9334649

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461
1: -576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391
2: -492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311
3: -691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342
4: -652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
time: 0.75 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.06
Output dim: 0, lower bound: -1541.9242151, upper bound: 1541.9242151

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.31 + 71.82 = 75.13 seconds
