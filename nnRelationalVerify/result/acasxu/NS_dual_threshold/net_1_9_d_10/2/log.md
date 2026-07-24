## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 7.420799999999999e-05


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0202495, -0.0201211, -0.0202495, -0.0201211, -0.0001284, 0.0001284)
1: (-0.0191812, -0.0189370, -0.0191812, -0.0189370, -0.0002442, 0.0002442)
2: (-0.0191748, -0.0188630, -0.0191748, -0.0188630, -0.0003118, 0.0003118)
3: (-0.0182793, -0.0179404, -0.0182793, -0.0179404, -0.0003389, 0.0003389)
4: (-0.0184249, -0.0178994, -0.0184249, -0.0178994, -0.0005255, 0.0005255)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.94 + 0.53 = 1.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000773, upper bound: 0.0000773

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000770
time: 0.12 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000771, upper bound: 0.0000771
time: 0.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.34 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.34
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000770
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.34
Output dim: 0, lower bound: -0.0000771, upper bound: 0.0000771

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0202572, -0.0201272, -0.0202485, -0.0201219, -0.0001353, 0.0001213
1: -0.0191468, -0.0189056, -0.0191789, -0.0189394, -0.0002074, 0.0002733
2: -0.0191347, -0.0188588, -0.0191721, -0.0188661, -0.0002686, 0.0003133
3: -0.0182448, -0.0179346, -0.0182768, -0.0179436, -0.0003012, 0.0003422
4: -0.0183621, -0.0179328, -0.0184204, -0.0179049, -0.0004571, 0.0004876

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000745, upper bound: 0.0000767
time: 0.14 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000757, upper bound: 0.0000769
time: 0.13 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0202451, -0.0201263, -0.0202495, -0.0201211, -0.0001240, 0.0001232
1: -0.0191720, -0.0189471, -0.0191812, -0.0189370, -0.0002350, 0.0002341
2: -0.0191657, -0.0188740, -0.0191748, -0.0188630, -0.0003027, 0.0003008
3: -0.0182697, -0.0179544, -0.0182793, -0.0179404, -0.0003293, 0.0003249
4: -0.0184107, -0.0179159, -0.0184249, -0.0178994, -0.0005113, 0.0005090

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000758
time: 0.15 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000758
time: 0.13 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.23 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0000745, upper bound: 0.0000767
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0000757, upper bound: 0.0000769
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000758
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000758

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.0202709, -0.0201282, -0.0202485, -0.0201219, -0.0001489, 0.0001203
1: -0.0191539, -0.0188836, -0.0191789, -0.0189394, -0.0002145, 0.0002954
2: -0.0191498, -0.0188484, -0.0191721, -0.0188661, -0.0002837, 0.0003236
3: -0.0182547, -0.0179366, -0.0182768, -0.0179436, -0.0003111, 0.0003402
4: -0.0183871, -0.0179436, -0.0184204, -0.0179049, -0.0004822, 0.0004768

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000761
time: 0.13 seconds

## Relational analysis of NS_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000758
time: 0.14 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000758
time: 0.14 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.0202567, -0.0201297, -0.0202485, -0.0201219, -0.0001348, 0.0001188
1: -0.0191466, -0.0189068, -0.0191789, -0.0189394, -0.0002072, 0.0002721
2: -0.0191346, -0.0188603, -0.0191721, -0.0188661, -0.0002685, 0.0003118
3: -0.0182447, -0.0179372, -0.0182768, -0.0179436, -0.0003011, 0.0003396
4: -0.0183619, -0.0179363, -0.0184204, -0.0179049, -0.0004570, 0.0004841

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000756
time: 0.13 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000769
time: 0.13 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202451, -0.0201263, -0.0202572, -0.0201272, -0.0001179, 0.0001309
1: -0.0191720, -0.0189471, -0.0191468, -0.0189056, -0.0002663, 0.0001997
2: -0.0191657, -0.0188740, -0.0191347, -0.0188588, -0.0003069, 0.0002607
3: -0.0182697, -0.0179544, -0.0182448, -0.0179346, -0.0003351, 0.0002904
4: -0.0184107, -0.0179159, -0.0183621, -0.0179328, -0.0004779, 0.0004462

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000742
time: 0.13 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000757
time: 0.13 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202451, -0.0201263, -0.0202451, -0.0201263, -0.0001187, 0.0001187
1: -0.0191720, -0.0189471, -0.0191720, -0.0189471, -0.0002249, 0.0002249
2: -0.0191657, -0.0188740, -0.0191657, -0.0188740, -0.0002917, 0.0002917
3: -0.0182697, -0.0179544, -0.0182697, -0.0179544, -0.0003153, 0.0003153
4: -0.0184107, -0.0179159, -0.0184107, -0.0179159, -0.0004948, 0.0004948

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000767
time: 0.14 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000757
time: 0.16 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.64 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000758
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000758
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000756
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000769
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000742
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000757
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000767
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000757

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202709, -0.0201282, -0.0202572, -0.0201272, -0.0001437, 0.0001290
1: -0.0191539, -0.0188836, -0.0191468, -0.0189056, -0.0002483, 0.0002632
2: -0.0191498, -0.0188484, -0.0191347, -0.0188588, -0.0002910, 0.0002863
3: -0.0182547, -0.0179366, -0.0182448, -0.0179346, -0.0003202, 0.0003082
4: -0.0183871, -0.0179436, -0.0183621, -0.0179328, -0.0004543, 0.0004185

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000743
time: 0.15 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000758
time: 0.15 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202709, -0.0201282, -0.0202451, -0.0201263, -0.0001445, 0.0001169
1: -0.0191539, -0.0188836, -0.0191720, -0.0189471, -0.0002068, 0.0002884
2: -0.0191498, -0.0188484, -0.0191657, -0.0188740, -0.0002757, 0.0003173
3: -0.0182547, -0.0179366, -0.0182697, -0.0179544, -0.0003004, 0.0003331
4: -0.0183871, -0.0179436, -0.0184107, -0.0179159, -0.0004712, 0.0004671

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000743
time: 0.16 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000758
time: 0.16 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202567, -0.0201297, -0.0202572, -0.0201272, -0.0001295, 0.0001276
1: -0.0191466, -0.0189068, -0.0191468, -0.0189056, -0.0002410, 0.0002400
2: -0.0191346, -0.0188603, -0.0191347, -0.0188588, -0.0002758, 0.0002744
3: -0.0182447, -0.0179372, -0.0182448, -0.0179346, -0.0003101, 0.0003076
4: -0.0183619, -0.0179363, -0.0183621, -0.0179328, -0.0004291, 0.0004258

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000754
time: 0.15 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
time: 0.14 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202567, -0.0201297, -0.0202451, -0.0201263, -0.0001304, 0.0001154
1: -0.0191466, -0.0189068, -0.0191720, -0.0189471, -0.0001995, 0.0002651
2: -0.0191346, -0.0188603, -0.0191657, -0.0188740, -0.0002605, 0.0003054
3: -0.0182447, -0.0179372, -0.0182697, -0.0179544, -0.0002903, 0.0003325
4: -0.0183619, -0.0179363, -0.0184107, -0.0179159, -0.0004460, 0.0004744

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000754
time: 0.13 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
time: 0.14 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0202451, -0.0201263, -0.0202709, -0.0201282, -0.0001169, 0.0001445
1: -0.0191720, -0.0189471, -0.0191539, -0.0188836, -0.0002884, 0.0002068
2: -0.0191657, -0.0188740, -0.0191498, -0.0188484, -0.0003173, 0.0002757
3: -0.0182697, -0.0179544, -0.0182547, -0.0179366, -0.0003331, 0.0003004
4: -0.0184107, -0.0179159, -0.0183871, -0.0179436, -0.0004671, 0.0004712

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000742
time: 0.15 seconds

## Relational analysis of NS_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000735
time: 0.15 seconds

## Relational analysis of NS_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000745
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000745
time: 0.14 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202451, -0.0201263, -0.0202567, -0.0201297, -0.0001154, 0.0001304
1: -0.0191720, -0.0189471, -0.0191466, -0.0189068, -0.0002651, 0.0001995
2: -0.0191657, -0.0188740, -0.0191346, -0.0188603, -0.0003054, 0.0002605
3: -0.0182697, -0.0179544, -0.0182447, -0.0179372, -0.0003325, 0.0002903
4: -0.0184107, -0.0179159, -0.0183619, -0.0179363, -0.0004744, 0.0004460

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000754
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000748
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000750
time: 0.14 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000750
time: 0.15 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202591, -0.0201288, -0.0202451, -0.0201263, -0.0001327, 0.0001163
1: -0.0191786, -0.0189208, -0.0191720, -0.0189471, -0.0002315, 0.0002512
2: -0.0191804, -0.0188584, -0.0191657, -0.0188740, -0.0003064, 0.0003073
3: -0.0182787, -0.0179540, -0.0182697, -0.0179544, -0.0003243, 0.0003157
4: -0.0184354, -0.0179177, -0.0184107, -0.0179159, -0.0005195, 0.0004930

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000728, upper bound: 0.0000750
time: 0.15 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000760
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202437, -0.0201281, -0.0202451, -0.0201263, -0.0001174, 0.0001169
1: -0.0191713, -0.0189505, -0.0191720, -0.0189471, -0.0002242, 0.0002215
2: -0.0191649, -0.0188776, -0.0191657, -0.0188740, -0.0002909, 0.0002881
3: -0.0182689, -0.0179586, -0.0182697, -0.0179544, -0.0003146, 0.0003111
4: -0.0184095, -0.0179205, -0.0184107, -0.0179159, -0.0004936, 0.0004902

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000742
time: 0.14 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000770
time: 0.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.29 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000743
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000758
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000743
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000758
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000754
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000754
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000745
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000740, upper bound: 0.0000745
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000750
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000750
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000728, upper bound: 0.0000750
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000760
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000742
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.29
Output dim: 0, lower bound: -0.0000767, upper bound: 0.0000770

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0202709, -0.0201282, -0.0202709, -0.0201282, -0.0001427, 0.0001427
1: -0.0191539, -0.0188836, -0.0191539, -0.0188836, -0.0002703, 0.0002703
2: -0.0191498, -0.0188484, -0.0191498, -0.0188484, -0.0003013, 0.0003013
3: -0.0182547, -0.0179366, -0.0182547, -0.0179366, -0.0003182, 0.0003182
4: -0.0183871, -0.0179436, -0.0183871, -0.0179436, -0.0004435, 0.0004435

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202709, -0.0201282, -0.0202567, -0.0201297, -0.0001412, 0.0001285
1: -0.0191539, -0.0188836, -0.0191466, -0.0189068, -0.0002471, 0.0002631
2: -0.0191498, -0.0188484, -0.0191346, -0.0188603, -0.0002895, 0.0002861
3: -0.0182547, -0.0179366, -0.0182447, -0.0179372, -0.0003176, 0.0003081
4: -0.0183871, -0.0179436, -0.0183619, -0.0179363, -0.0004508, 0.0004183

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0202709, -0.0201282, -0.0202591, -0.0201288, -0.0001421, 0.0001309
1: -0.0191539, -0.0188836, -0.0191786, -0.0189208, -0.0002331, 0.0002950
2: -0.0191498, -0.0188484, -0.0191804, -0.0188584, -0.0002914, 0.0003320
3: -0.0182547, -0.0179366, -0.0182787, -0.0179540, -0.0003008, 0.0003421
4: -0.0183871, -0.0179436, -0.0184354, -0.0179177, -0.0004694, 0.0004918

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202709, -0.0201282, -0.0202437, -0.0201281, -0.0001427, 0.0001155
1: -0.0191539, -0.0188836, -0.0191713, -0.0189505, -0.0002034, 0.0002877
2: -0.0191498, -0.0188484, -0.0191649, -0.0188776, -0.0002722, 0.0003165
3: -0.0182547, -0.0179366, -0.0182689, -0.0179586, -0.0002961, 0.0003323
4: -0.0183871, -0.0179436, -0.0184095, -0.0179205, -0.0004666, 0.0004659

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0202567, -0.0201297, -0.0202564, -0.0201275, -0.0001292, 0.0001267
1: -0.0191466, -0.0189068, -0.0191428, -0.0189077, -0.0002389, 0.0002360
2: -0.0191346, -0.0188603, -0.0191301, -0.0188604, -0.0002742, 0.0002698
3: -0.0182447, -0.0179372, -0.0182394, -0.0179442, -0.0003005, 0.0003022
4: -0.0183619, -0.0179363, -0.0183554, -0.0179394, -0.0004225, 0.0004192

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
time: 0.16 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
time: 0.13 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202567, -0.0201297, -0.0202560, -0.0201275, -0.0001292, 0.0001263
1: -0.0191466, -0.0189068, -0.0191450, -0.0189089, -0.0002378, 0.0002382
2: -0.0191346, -0.0188603, -0.0191322, -0.0188625, -0.0002720, 0.0002719
3: -0.0182447, -0.0179372, -0.0182425, -0.0179389, -0.0003058, 0.0003053
4: -0.0183619, -0.0179363, -0.0183584, -0.0179394, -0.0004225, 0.0004221

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
time: 0.15 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
time: 0.14 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0202567, -0.0201297, -0.0202477, -0.0201265, -0.0001302, 0.0001180
1: -0.0191466, -0.0189068, -0.0191761, -0.0189385, -0.0002081, 0.0002692
2: -0.0191346, -0.0188603, -0.0191738, -0.0188618, -0.0002727, 0.0003135
3: -0.0182447, -0.0179372, -0.0182729, -0.0179470, -0.0002977, 0.0003357
4: -0.0183619, -0.0179363, -0.0184248, -0.0178953, -0.0004666, 0.0004885

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000763
time: 0.15 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000763
time: 0.17 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202567, -0.0201297, -0.0202419, -0.0201270, -0.0001297, 0.0001123
1: -0.0191466, -0.0189068, -0.0191693, -0.0189557, -0.0001910, 0.0002625
2: -0.0191346, -0.0188603, -0.0191623, -0.0188831, -0.0002514, 0.0003020
3: -0.0182447, -0.0179372, -0.0182661, -0.0179650, -0.0002797, 0.0003289
4: -0.0183619, -0.0179363, -0.0184059, -0.0179292, -0.0004327, 0.0004697

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000766
time: 0.15 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000767
time: 0.15 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202591, -0.0201288, -0.0202709, -0.0201282, -0.0001309, 0.0001421
1: -0.0191786, -0.0189208, -0.0191539, -0.0188836, -0.0002950, 0.0002331
2: -0.0191804, -0.0188584, -0.0191498, -0.0188484, -0.0003320, 0.0002914
3: -0.0182787, -0.0179540, -0.0182547, -0.0179366, -0.0003421, 0.0003008
4: -0.0184354, -0.0179177, -0.0183871, -0.0179436, -0.0004918, 0.0004694

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202437, -0.0201281, -0.0202709, -0.0201282, -0.0001155, 0.0001427
1: -0.0191713, -0.0189505, -0.0191539, -0.0188836, -0.0002877, 0.0002034
2: -0.0191649, -0.0188776, -0.0191498, -0.0188484, -0.0003165, 0.0002722
3: -0.0182689, -0.0179586, -0.0182547, -0.0179366, -0.0003323, 0.0002961
4: -0.0184095, -0.0179205, -0.0183871, -0.0179436, -0.0004659, 0.0004666

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202477, -0.0201265, -0.0202567, -0.0201297, -0.0001180, 0.0001302
1: -0.0191761, -0.0189385, -0.0191466, -0.0189068, -0.0002692, 0.0002081
2: -0.0191738, -0.0188618, -0.0191346, -0.0188603, -0.0003135, 0.0002727
3: -0.0182729, -0.0179470, -0.0182447, -0.0179372, -0.0003357, 0.0002977
4: -0.0184248, -0.0178953, -0.0183619, -0.0179363, -0.0004885, 0.0004666

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000750
time: 0.14 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000750
time: 0.15 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202419, -0.0201270, -0.0202567, -0.0201297, -0.0001123, 0.0001297
1: -0.0191693, -0.0189557, -0.0191466, -0.0189068, -0.0002625, 0.0001910
2: -0.0191623, -0.0188831, -0.0191346, -0.0188603, -0.0003020, 0.0002514
3: -0.0182661, -0.0179650, -0.0182447, -0.0179372, -0.0003289, 0.0002797
4: -0.0184059, -0.0179292, -0.0183619, -0.0179363, -0.0004697, 0.0004327

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000751
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000750
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0202549, -0.0201221, -0.0202451, -0.0201263, -0.0001285, 0.0001229
1: -0.0191806, -0.0189291, -0.0191720, -0.0189471, -0.0002335, 0.0002429
2: -0.0191825, -0.0188755, -0.0191657, -0.0188740, -0.0003085, 0.0002902
3: -0.0182808, -0.0179631, -0.0182697, -0.0179544, -0.0003265, 0.0003066
4: -0.0184391, -0.0179460, -0.0184107, -0.0179159, -0.0005232, 0.0004647

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000719, upper bound: 0.0000726
time: 0.16 seconds

## Relational analysis of NS_A2_B2_A1_A1_A2

### Relational analysis result of NS_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0202514, -0.0201299, -0.0202451, -0.0201263, -0.0001250, 0.0001152
1: -0.0191755, -0.0189373, -0.0191720, -0.0189471, -0.0002284, 0.0002347
2: -0.0191769, -0.0188723, -0.0191657, -0.0188740, -0.0003028, 0.0002934
3: -0.0182753, -0.0179696, -0.0182697, -0.0179544, -0.0003209, 0.0003001
4: -0.0184303, -0.0179365, -0.0184107, -0.0179159, -0.0005144, 0.0004742

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000754
time: 0.16 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000759
time: 0.15 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202437, -0.0201281, -0.0202591, -0.0201288, -0.0001150, 0.0001309
1: -0.0191713, -0.0189505, -0.0191786, -0.0189208, -0.0002505, 0.0002281
2: -0.0191649, -0.0188776, -0.0191804, -0.0188584, -0.0003065, 0.0003029
3: -0.0182689, -0.0179586, -0.0182787, -0.0179540, -0.0003150, 0.0003201
4: -0.0184095, -0.0179205, -0.0184354, -0.0179177, -0.0004918, 0.0005149

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000739
time: 0.16 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000725
time: 0.15 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202437, -0.0201281, -0.0202437, -0.0201281, -0.0001156, 0.0001156
1: -0.0191713, -0.0189505, -0.0191713, -0.0189505, -0.0002208, 0.0002208
2: -0.0191649, -0.0188776, -0.0191649, -0.0188776, -0.0002873, 0.0002873
3: -0.0182689, -0.0179586, -0.0182689, -0.0179586, -0.0003103, 0.0003103
4: -0.0184095, -0.0179205, -0.0184095, -0.0179205, -0.0004890, 0.0004890

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000754
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000757
time: 0.18 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.46 seconds
NS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
NS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
NS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
NS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000748
NS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000763
NS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000763
NS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000766
NS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000751, upper bound: 0.0000767
NS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000750
NS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000750
NS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000751
NS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000750
NS_A2_B2_A1_A1_A1, status: Status.VERIFIED, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000719, upper bound: 0.0000726
NS_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
NS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000754
NS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000759
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000739
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000725
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000754
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.46
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000757

## BFS NS instance: NS_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202557, -0.0201300, -0.0202564, -0.0201275, -0.0001282, 0.0001264
1: -0.0191426, -0.0189093, -0.0191428, -0.0189077, -0.0002349, 0.0002335
2: -0.0191299, -0.0188622, -0.0191301, -0.0188604, -0.0002695, 0.0002678
3: -0.0182393, -0.0179475, -0.0182394, -0.0179442, -0.0002951, 0.0002920
4: -0.0183552, -0.0179424, -0.0183554, -0.0179394, -0.0004158, 0.0004130

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B1_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000741
time: 0.16 seconds

## Relational analysis of NS_A1_A2_B1_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000745
time: 0.15 seconds

## BFS NS instance: NS_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202554, -0.0201300, -0.0202564, -0.0201275, -0.0001279, 0.0001264
1: -0.0191449, -0.0189101, -0.0191428, -0.0189077, -0.0002371, 0.0002327
2: -0.0191320, -0.0188641, -0.0191301, -0.0188604, -0.0002716, 0.0002660
3: -0.0182424, -0.0179416, -0.0182394, -0.0179442, -0.0002982, 0.0002978
4: -0.0183582, -0.0179430, -0.0183554, -0.0179394, -0.0004188, 0.0004124

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B1_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000741
time: 0.16 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000745
time: 0.15 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202557, -0.0201300, -0.0202560, -0.0201275, -0.0001282, 0.0001260
1: -0.0191426, -0.0189093, -0.0191450, -0.0189089, -0.0002338, 0.0002358
2: -0.0191299, -0.0188622, -0.0191322, -0.0188625, -0.0002674, 0.0002699
3: -0.0182393, -0.0179475, -0.0182425, -0.0179389, -0.0003004, 0.0002950
4: -0.0183552, -0.0179424, -0.0183584, -0.0179394, -0.0004158, 0.0004160

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202554, -0.0201300, -0.0202560, -0.0201275, -0.0001279, 0.0001260
1: -0.0191449, -0.0189101, -0.0191450, -0.0189089, -0.0002360, 0.0002349
2: -0.0191320, -0.0188641, -0.0191322, -0.0188625, -0.0002695, 0.0002681
3: -0.0182424, -0.0179416, -0.0182425, -0.0179389, -0.0003035, 0.0003009
4: -0.0183582, -0.0179430, -0.0183584, -0.0179394, -0.0004188, 0.0004153

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202557, -0.0201300, -0.0202477, -0.0201265, -0.0001292, 0.0001176
1: -0.0191426, -0.0189093, -0.0191761, -0.0189385, -0.0002041, 0.0002668
2: -0.0191299, -0.0188622, -0.0191738, -0.0188618, -0.0002681, 0.0003115
3: -0.0182393, -0.0179475, -0.0182729, -0.0179470, -0.0002923, 0.0003254
4: -0.0183552, -0.0179424, -0.0184248, -0.0178953, -0.0004599, 0.0004824

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000738
time: 0.17 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000746
time: 0.17 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202554, -0.0201300, -0.0202477, -0.0201265, -0.0001289, 0.0001176
1: -0.0191449, -0.0189101, -0.0191761, -0.0189385, -0.0002064, 0.0002659
2: -0.0191320, -0.0188641, -0.0191738, -0.0188618, -0.0002702, 0.0003097
3: -0.0182424, -0.0179416, -0.0182729, -0.0179470, -0.0002954, 0.0003313
4: -0.0183582, -0.0179430, -0.0184248, -0.0178953, -0.0004629, 0.0004818

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000738
time: 0.17 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000746
time: 0.17 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202557, -0.0201300, -0.0202419, -0.0201270, -0.0001288, 0.0001119
1: -0.0191426, -0.0189093, -0.0191693, -0.0189557, -0.0001869, 0.0002601
2: -0.0191299, -0.0188622, -0.0191623, -0.0188831, -0.0002468, 0.0003000
3: -0.0182393, -0.0179475, -0.0182661, -0.0179650, -0.0002744, 0.0003186
4: -0.0183552, -0.0179424, -0.0184059, -0.0179292, -0.0004261, 0.0004636

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000737
time: 0.18 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000746
time: 0.18 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202554, -0.0201300, -0.0202419, -0.0201270, -0.0001285, 0.0001119
1: -0.0191449, -0.0189101, -0.0191693, -0.0189557, -0.0001892, 0.0002592
2: -0.0191320, -0.0188641, -0.0191623, -0.0188831, -0.0002489, 0.0002982
3: -0.0182424, -0.0179416, -0.0182661, -0.0179650, -0.0002774, 0.0003245
4: -0.0183582, -0.0179430, -0.0184059, -0.0179292, -0.0004290, 0.0004629

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000737
time: 0.19 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000746
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202477, -0.0201265, -0.0202557, -0.0201300, -0.0001176, 0.0001292
1: -0.0191761, -0.0189385, -0.0191426, -0.0189093, -0.0002668, 0.0002041
2: -0.0191738, -0.0188618, -0.0191299, -0.0188622, -0.0003115, 0.0002681
3: -0.0182729, -0.0179470, -0.0182393, -0.0179475, -0.0003254, 0.0002923
4: -0.0184248, -0.0178953, -0.0183552, -0.0179424, -0.0004824, 0.0004599

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202477, -0.0201265, -0.0202554, -0.0201300, -0.0001176, 0.0001289
1: -0.0191761, -0.0189385, -0.0191449, -0.0189101, -0.0002659, 0.0002064
2: -0.0191738, -0.0188618, -0.0191320, -0.0188641, -0.0003097, 0.0002702
3: -0.0182729, -0.0179470, -0.0182424, -0.0179416, -0.0003313, 0.0002954
4: -0.0184248, -0.0178953, -0.0183582, -0.0179430, -0.0004818, 0.0004629

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202419, -0.0201270, -0.0202557, -0.0201300, -0.0001119, 0.0001288
1: -0.0191693, -0.0189557, -0.0191426, -0.0189093, -0.0002601, 0.0001869
2: -0.0191623, -0.0188831, -0.0191299, -0.0188622, -0.0003000, 0.0002468
3: -0.0182661, -0.0179650, -0.0182393, -0.0179475, -0.0003186, 0.0002744
4: -0.0184059, -0.0179292, -0.0183552, -0.0179424, -0.0004636, 0.0004261

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202419, -0.0201270, -0.0202554, -0.0201300, -0.0001119, 0.0001285
1: -0.0191693, -0.0189557, -0.0191449, -0.0189101, -0.0002592, 0.0001892
2: -0.0191623, -0.0188831, -0.0191320, -0.0188641, -0.0002982, 0.0002489
3: -0.0182661, -0.0179650, -0.0182424, -0.0179416, -0.0003245, 0.0002774
4: -0.0184059, -0.0179292, -0.0183582, -0.0179430, -0.0004629, 0.0004290

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202451, -0.0201263, -0.0001265, 0.0001224
1: -0.0191780, -0.0189334, -0.0191720, -0.0189471, -0.0002309, 0.0002385
2: -0.0191794, -0.0188810, -0.0191657, -0.0188740, -0.0003053, 0.0002847
3: -0.0182772, -0.0179695, -0.0182697, -0.0179544, -0.0003228, 0.0003002
4: -0.0184346, -0.0179542, -0.0184107, -0.0179159, -0.0005187, 0.0004565

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
time: 0.16 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202514, -0.0201299, -0.0202477, -0.0201265, -0.0001249, 0.0001178
1: -0.0191755, -0.0189373, -0.0191761, -0.0189385, -0.0002370, 0.0002388
2: -0.0191769, -0.0188723, -0.0191738, -0.0188618, -0.0003151, 0.0003014
3: -0.0182753, -0.0179696, -0.0182729, -0.0179470, -0.0003283, 0.0003033
4: -0.0184303, -0.0179365, -0.0184248, -0.0178953, -0.0005350, 0.0004884

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000754
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202514, -0.0201299, -0.0202419, -0.0201270, -0.0001244, 0.0001120
1: -0.0191755, -0.0189373, -0.0191693, -0.0189557, -0.0002199, 0.0002321
2: -0.0191769, -0.0188723, -0.0191623, -0.0188831, -0.0002938, 0.0002899
3: -0.0182753, -0.0179696, -0.0182661, -0.0179650, -0.0003103, 0.0002965
4: -0.0184303, -0.0179365, -0.0184059, -0.0179292, -0.0005011, 0.0004695

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000759
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0202437, -0.0201281, -0.0202549, -0.0201221, -0.0001216, 0.0001267
1: -0.0191713, -0.0189505, -0.0191806, -0.0189291, -0.0002422, 0.0002301
2: -0.0191649, -0.0188776, -0.0191825, -0.0188755, -0.0002894, 0.0003050
3: -0.0182689, -0.0179586, -0.0182808, -0.0179631, -0.0003058, 0.0003222
4: -0.0184095, -0.0179205, -0.0184391, -0.0179460, -0.0004636, 0.0005185

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000738
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000736
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202437, -0.0201281, -0.0202514, -0.0201299, -0.0001139, 0.0001232
1: -0.0191713, -0.0189505, -0.0191755, -0.0189373, -0.0002340, 0.0002251
2: -0.0191649, -0.0188776, -0.0191769, -0.0188723, -0.0002926, 0.0002993
3: -0.0182689, -0.0179586, -0.0182753, -0.0179696, -0.0002994, 0.0003167
4: -0.0184095, -0.0179205, -0.0184303, -0.0179365, -0.0004731, 0.0005097

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000722
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000724
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202462, -0.0201283, -0.0202437, -0.0201281, -0.0001181, 0.0001154
1: -0.0191755, -0.0189421, -0.0191713, -0.0189505, -0.0002250, 0.0002292
2: -0.0191731, -0.0188655, -0.0191649, -0.0188776, -0.0002955, 0.0002994
3: -0.0182722, -0.0179506, -0.0182689, -0.0179586, -0.0003136, 0.0003183
4: -0.0184238, -0.0179001, -0.0184095, -0.0179205, -0.0005033, 0.0005095

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000737
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000747
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202405, -0.0201287, -0.0202437, -0.0201281, -0.0001123, 0.0001150
1: -0.0191687, -0.0189593, -0.0191713, -0.0189505, -0.0002182, 0.0002120
2: -0.0191614, -0.0188869, -0.0191649, -0.0188776, -0.0002839, 0.0002779
3: -0.0182653, -0.0179696, -0.0182689, -0.0179586, -0.0003067, 0.0002994
4: -0.0184047, -0.0179342, -0.0184095, -0.0179205, -0.0004842, 0.0004754

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000753
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000751
time: 0.17 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.53 seconds
NS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000741
NS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000745
NS_A1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000741
NS_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000745
NS_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000738
NS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000746
NS_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000738
NS_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000746
NS_A1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000737
NS_A1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000746
NS_A1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000737
NS_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000750, upper bound: 0.0000746
NS_A2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
NS_A2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
NS_A2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
NS_A2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000754
NS_A2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
NS_A2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000759
NS_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000738
NS_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000736
NS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000722
NS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000724
NS_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000758, upper bound: 0.0000737
NS_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000747
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000753
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.53
Output dim: 0, lower bound: -0.0000766, upper bound: 0.0000751

## BFS NS instance: NS_A1_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202557, -0.0201300, -0.0202714, -0.0201286, -0.0001271, 0.0001414
1: -0.0191426, -0.0189093, -0.0191496, -0.0188807, -0.0002619, 0.0002404
2: -0.0191299, -0.0188622, -0.0191448, -0.0188441, -0.0002858, 0.0002825
3: -0.0182393, -0.0179475, -0.0182492, -0.0179314, -0.0003080, 0.0003017
4: -0.0183552, -0.0179424, -0.0183802, -0.0179376, -0.0004176, 0.0004378

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202557, -0.0201300, -0.0202557, -0.0201300, -0.0001257, 0.0001257
1: -0.0191426, -0.0189093, -0.0191426, -0.0189093, -0.0002333, 0.0002333
2: -0.0191299, -0.0188622, -0.0191299, -0.0188622, -0.0002677, 0.0002677
3: -0.0182393, -0.0179475, -0.0182393, -0.0179475, -0.0002919, 0.0002919
4: -0.0183552, -0.0179424, -0.0183552, -0.0179424, -0.0004128, 0.0004128

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202554, -0.0201300, -0.0202714, -0.0201286, -0.0001268, 0.0001414
1: -0.0191449, -0.0189101, -0.0191496, -0.0188807, -0.0002641, 0.0002395
2: -0.0191320, -0.0188641, -0.0191448, -0.0188441, -0.0002880, 0.0002807
3: -0.0182424, -0.0179416, -0.0182492, -0.0179314, -0.0003110, 0.0003076
4: -0.0183582, -0.0179430, -0.0183802, -0.0179376, -0.0004205, 0.0004372

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202554, -0.0201300, -0.0202557, -0.0201300, -0.0001254, 0.0001257
1: -0.0191449, -0.0189101, -0.0191426, -0.0189093, -0.0002356, 0.0002325
2: -0.0191320, -0.0188641, -0.0191299, -0.0188622, -0.0002698, 0.0002658
3: -0.0182424, -0.0179416, -0.0182393, -0.0179475, -0.0002950, 0.0002977
4: -0.0183582, -0.0179430, -0.0183552, -0.0179424, -0.0004158, 0.0004122

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202557, -0.0201300, -0.0202598, -0.0201301, -0.0001257, 0.0001297
1: -0.0191426, -0.0189093, -0.0191813, -0.0189178, -0.0002249, 0.0002721
2: -0.0191299, -0.0188622, -0.0191861, -0.0188498, -0.0002800, 0.0003239
3: -0.0182393, -0.0179475, -0.0182805, -0.0179438, -0.0002956, 0.0003331
4: -0.0183552, -0.0179424, -0.0184454, -0.0179036, -0.0004516, 0.0005030

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202557, -0.0201300, -0.0202462, -0.0201283, -0.0001274, 0.0001162
1: -0.0191426, -0.0189093, -0.0191755, -0.0189421, -0.0002005, 0.0002662
2: -0.0191299, -0.0188622, -0.0191731, -0.0188655, -0.0002644, 0.0003108
3: -0.0182393, -0.0179475, -0.0182722, -0.0179506, -0.0002887, 0.0003248
4: -0.0183552, -0.0179424, -0.0184238, -0.0179001, -0.0004552, 0.0004814

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202554, -0.0201300, -0.0202598, -0.0201301, -0.0001254, 0.0001298
1: -0.0191449, -0.0189101, -0.0191813, -0.0189178, -0.0002271, 0.0002712
2: -0.0191320, -0.0188641, -0.0191861, -0.0188498, -0.0002822, 0.0003221
3: -0.0182424, -0.0179416, -0.0182805, -0.0179438, -0.0002986, 0.0003389
4: -0.0183582, -0.0179430, -0.0184454, -0.0179036, -0.0004546, 0.0005023

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202554, -0.0201300, -0.0202462, -0.0201283, -0.0001271, 0.0001162
1: -0.0191449, -0.0189101, -0.0191755, -0.0189421, -0.0002028, 0.0002654
2: -0.0191320, -0.0188641, -0.0191731, -0.0188655, -0.0002665, 0.0003090
3: -0.0182424, -0.0179416, -0.0182722, -0.0179506, -0.0002918, 0.0003306
4: -0.0183582, -0.0179430, -0.0184238, -0.0179001, -0.0004581, 0.0004808

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202557, -0.0201300, -0.0202566, -0.0201292, -0.0001266, 0.0001266
1: -0.0191426, -0.0189093, -0.0191762, -0.0189265, -0.0002161, 0.0002669
2: -0.0191299, -0.0188622, -0.0191774, -0.0188658, -0.0002641, 0.0003151
3: -0.0182393, -0.0179475, -0.0182754, -0.0179612, -0.0002782, 0.0003279
4: -0.0183552, -0.0179424, -0.0184311, -0.0179291, -0.0004262, 0.0004887

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202557, -0.0201300, -0.0202405, -0.0201287, -0.0001270, 0.0001104
1: -0.0191426, -0.0189093, -0.0191687, -0.0189593, -0.0001833, 0.0002594
2: -0.0191299, -0.0188622, -0.0191614, -0.0188869, -0.0002429, 0.0002992
3: -0.0182393, -0.0179475, -0.0182653, -0.0179696, -0.0002698, 0.0003179
4: -0.0183552, -0.0179424, -0.0184047, -0.0179342, -0.0004211, 0.0004623

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202554, -0.0201300, -0.0202566, -0.0201292, -0.0001263, 0.0001266
1: -0.0191449, -0.0189101, -0.0191762, -0.0189265, -0.0002183, 0.0002660
2: -0.0191320, -0.0188641, -0.0191774, -0.0188658, -0.0002663, 0.0003133
3: -0.0182424, -0.0179416, -0.0182754, -0.0179612, -0.0002812, 0.0003338
4: -0.0183582, -0.0179430, -0.0184311, -0.0179291, -0.0004291, 0.0004881

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A1_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202554, -0.0201300, -0.0202405, -0.0201287, -0.0001267, 0.0001105
1: -0.0191449, -0.0189101, -0.0191687, -0.0189593, -0.0001855, 0.0002585
2: -0.0191320, -0.0188641, -0.0191614, -0.0188869, -0.0002451, 0.0002974
3: -0.0182424, -0.0179416, -0.0182653, -0.0179696, -0.0002728, 0.0003237
4: -0.0183582, -0.0179430, -0.0184047, -0.0179342, -0.0004240, 0.0004617

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## BFS NS instance: NS_A2_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202477, -0.0201265, -0.0001263, 0.0001250
1: -0.0191780, -0.0189334, -0.0191761, -0.0189385, -0.0002395, 0.0002426
2: -0.0191794, -0.0188810, -0.0191738, -0.0188618, -0.0003175, 0.0002927
3: -0.0182772, -0.0179695, -0.0182729, -0.0179470, -0.0003302, 0.0003034
4: -0.0184346, -0.0179542, -0.0184248, -0.0178953, -0.0005393, 0.0004706

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_A1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202419, -0.0201270, -0.0001258, 0.0001192
1: -0.0191780, -0.0189334, -0.0191693, -0.0189557, -0.0002223, 0.0002359
2: -0.0191794, -0.0188810, -0.0191623, -0.0188831, -0.0002962, 0.0002812
3: -0.0182772, -0.0179695, -0.0182661, -0.0179650, -0.0003123, 0.0002966
4: -0.0184346, -0.0179542, -0.0184059, -0.0179292, -0.0005054, 0.0004517

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202546, -0.0201285, -0.0202477, -0.0201265, -0.0001281, 0.0001192
1: -0.0191814, -0.0189257, -0.0191761, -0.0189385, -0.0002429, 0.0002503
2: -0.0191859, -0.0188575, -0.0191738, -0.0188618, -0.0003240, 0.0003163
3: -0.0182789, -0.0179524, -0.0182729, -0.0179470, -0.0003319, 0.0003204
4: -0.0184453, -0.0179129, -0.0184248, -0.0178953, -0.0005500, 0.0005119

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
time: 0.16 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202488, -0.0201303, -0.0202477, -0.0201265, -0.0001223, 0.0001173
1: -0.0191733, -0.0189426, -0.0191761, -0.0189385, -0.0002348, 0.0002335
2: -0.0191738, -0.0188793, -0.0191738, -0.0188618, -0.0003120, 0.0002944
3: -0.0182721, -0.0179763, -0.0182729, -0.0179470, -0.0003251, 0.0002966
4: -0.0184259, -0.0179472, -0.0184248, -0.0178953, -0.0005306, 0.0004776

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202546, -0.0201285, -0.0202419, -0.0201270, -0.0001277, 0.0001134
1: -0.0191814, -0.0189257, -0.0191693, -0.0189557, -0.0002257, 0.0002436
2: -0.0191859, -0.0188575, -0.0191623, -0.0188831, -0.0003027, 0.0003048
3: -0.0182789, -0.0179524, -0.0182661, -0.0179650, -0.0003139, 0.0003136
4: -0.0184453, -0.0179129, -0.0184059, -0.0179292, -0.0005162, 0.0004930

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202488, -0.0201303, -0.0202419, -0.0201270, -0.0001219, 0.0001116
1: -0.0191733, -0.0189426, -0.0191693, -0.0189557, -0.0002176, 0.0002268
2: -0.0191738, -0.0188793, -0.0191623, -0.0188831, -0.0002907, 0.0002829
3: -0.0182721, -0.0179763, -0.0182661, -0.0179650, -0.0003071, 0.0002898
4: -0.0184259, -0.0179472, -0.0184059, -0.0179292, -0.0004968, 0.0004587

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0202437, -0.0201281, -0.0202563, -0.0201201, -0.0001236, 0.0001282
1: -0.0191713, -0.0189505, -0.0191861, -0.0189257, -0.0002456, 0.0002356
2: -0.0191649, -0.0188776, -0.0191913, -0.0188647, -0.0003002, 0.0003138
3: -0.0182689, -0.0179586, -0.0182848, -0.0179524, -0.0003165, 0.0003262
4: -0.0184095, -0.0179205, -0.0184537, -0.0179286, -0.0004809, 0.0005332

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202437, -0.0201281, -0.0202528, -0.0201227, -0.0001210, 0.0001246
1: -0.0191713, -0.0189505, -0.0191780, -0.0189334, -0.0002379, 0.0002275
2: -0.0191649, -0.0188776, -0.0191794, -0.0188810, -0.0002839, 0.0003018
3: -0.0182689, -0.0179586, -0.0182772, -0.0179695, -0.0002995, 0.0003186
4: -0.0184095, -0.0179205, -0.0184346, -0.0179542, -0.0004553, 0.0005141

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000737
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202462, -0.0201283, -0.0202514, -0.0201299, -0.0001163, 0.0001231
1: -0.0191755, -0.0189421, -0.0191755, -0.0189373, -0.0002382, 0.0002334
2: -0.0191731, -0.0188655, -0.0191769, -0.0188723, -0.0003007, 0.0003113
3: -0.0182722, -0.0179506, -0.0182753, -0.0179696, -0.0003026, 0.0003247
4: -0.0184238, -0.0179001, -0.0184303, -0.0179365, -0.0004873, 0.0005302

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000718
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000718
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202405, -0.0201287, -0.0202514, -0.0201299, -0.0001106, 0.0001227
1: -0.0191687, -0.0189593, -0.0191755, -0.0189373, -0.0002314, 0.0002162
2: -0.0191614, -0.0188869, -0.0191769, -0.0188723, -0.0002891, 0.0002899
3: -0.0182653, -0.0179696, -0.0182753, -0.0179696, -0.0002957, 0.0003057
4: -0.0184047, -0.0179342, -0.0184303, -0.0179365, -0.0004683, 0.0004961

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000719
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000724
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202437, -0.0201281, -0.0001100, 0.0001227
1: -0.0191756, -0.0189626, -0.0191713, -0.0189505, -0.0002251, 0.0002087
2: -0.0191720, -0.0188924, -0.0191649, -0.0188776, -0.0002944, 0.0002725
3: -0.0182728, -0.0179693, -0.0182689, -0.0179586, -0.0003142, 0.0002996
4: -0.0184224, -0.0179407, -0.0184095, -0.0179205, -0.0005018, 0.0004688

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000737
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000737
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202437, -0.0201281, -0.0001101, 0.0001148
1: -0.0191733, -0.0189577, -0.0191713, -0.0189505, -0.0002229, 0.0002136
2: -0.0191702, -0.0188809, -0.0191649, -0.0188776, -0.0002927, 0.0002840
3: -0.0182697, -0.0179663, -0.0182689, -0.0179586, -0.0003111, 0.0003026
4: -0.0184198, -0.0179187, -0.0184095, -0.0179205, -0.0004992, 0.0004908

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000747
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000747
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202405, -0.0201287, -0.0202380, -0.0201210, -0.0001195, 0.0001092
1: -0.0191687, -0.0189593, -0.0191717, -0.0189639, -0.0002048, 0.0002124
2: -0.0191614, -0.0188869, -0.0191650, -0.0188997, -0.0002618, 0.0002780
3: -0.0182653, -0.0179696, -0.0182697, -0.0179730, -0.0002923, 0.0003002
4: -0.0184047, -0.0179342, -0.0184101, -0.0179563, -0.0004484, 0.0004759

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000751
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000751
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202405, -0.0201287, -0.0202343, -0.0201291, -0.0001114, 0.0001056
1: -0.0191687, -0.0189593, -0.0191682, -0.0189696, -0.0001991, 0.0002089
2: -0.0191614, -0.0188869, -0.0191613, -0.0188954, -0.0002660, 0.0002744
3: -0.0182653, -0.0179696, -0.0182656, -0.0179802, -0.0002852, 0.0002960
4: -0.0184047, -0.0179342, -0.0184042, -0.0179424, -0.0004624, 0.0004701

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000751
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000751
time: 0.19 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.56 seconds
NS_A2_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
NS_A2_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
NS_A2_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
NS_A2_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
NS_A2_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
NS_A2_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
NS_A2_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
NS_A2_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
NS_A2_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
NS_A2_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
NS_A2_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
NS_A2_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
NS_A2_B2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
NS_A2_B2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
NS_A2_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
NS_A2_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000737
NS_A2_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000718
NS_A2_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000718
NS_A2_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000719
NS_A2_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000724
NS_A2_B2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000737
NS_A2_B2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000737
NS_A2_B2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000747
NS_A2_B2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000747
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000751
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000751
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000751
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.56
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000751

## BFS NS instance: NS_A2_B2_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202399, -0.0201201, -0.0001327, 0.0001172
1: -0.0191780, -0.0189334, -0.0191762, -0.0189592, -0.0002188, 0.0002428
2: -0.0191794, -0.0188810, -0.0191729, -0.0188883, -0.0002911, 0.0002919
3: -0.0182772, -0.0179695, -0.0182735, -0.0179654, -0.0003118, 0.0003041
4: -0.0184346, -0.0179542, -0.0184238, -0.0179356, -0.0004990, 0.0004696

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A1_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000728, upper bound: 0.0000720
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_A1_A2_B1_B1_B2

### Relational analysis result of NS_A2_B2_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000728, upper bound: 0.0000749
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202394, -0.0201275, -0.0001252, 0.0001167
1: -0.0191780, -0.0189334, -0.0191736, -0.0189546, -0.0002234, 0.0002402
2: -0.0191794, -0.0188810, -0.0191706, -0.0188776, -0.0003018, 0.0002896
3: -0.0182772, -0.0179695, -0.0182700, -0.0179633, -0.0003140, 0.0003005
4: -0.0184346, -0.0179542, -0.0184203, -0.0179145, -0.0005202, 0.0004661

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000728, upper bound: 0.0000720
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_A1_A2_B1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000728, upper bound: 0.0000749
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202362, -0.0201202, -0.0001326, 0.0001135
1: -0.0191780, -0.0189334, -0.0191697, -0.0189688, -0.0002092, 0.0002362
2: -0.0191794, -0.0188810, -0.0191625, -0.0189050, -0.0002744, 0.0002814
3: -0.0182772, -0.0179695, -0.0182667, -0.0179784, -0.0002988, 0.0002972
4: -0.0184346, -0.0179542, -0.0184068, -0.0179647, -0.0004699, 0.0004526

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A1_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000724
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A1_A1_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202319, -0.0201282, -0.0001246, 0.0001092
1: -0.0191780, -0.0189334, -0.0191662, -0.0189758, -0.0002022, 0.0002328
2: -0.0191794, -0.0188810, -0.0191586, -0.0189027, -0.0002766, 0.0002776
3: -0.0182772, -0.0179695, -0.0182628, -0.0179878, -0.0002894, 0.0002933
4: -0.0184346, -0.0179542, -0.0184005, -0.0179535, -0.0004811, 0.0004463

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000724
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202546, -0.0201285, -0.0202399, -0.0201201, -0.0001345, 0.0001114
1: -0.0191814, -0.0189257, -0.0191762, -0.0189592, -0.0002222, 0.0002505
2: -0.0191859, -0.0188575, -0.0191729, -0.0188883, -0.0002976, 0.0003154
3: -0.0182789, -0.0179524, -0.0182735, -0.0179654, -0.0003135, 0.0003211
4: -0.0184453, -0.0179129, -0.0184238, -0.0179356, -0.0005097, 0.0005109

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000721
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
time: 0.16 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202546, -0.0201285, -0.0202394, -0.0201275, -0.0001271, 0.0001110
1: -0.0191814, -0.0189257, -0.0191736, -0.0189546, -0.0002268, 0.0002478
2: -0.0191859, -0.0188575, -0.0191706, -0.0188776, -0.0003083, 0.0003131
3: -0.0182789, -0.0179524, -0.0182700, -0.0179633, -0.0003156, 0.0003176
4: -0.0184453, -0.0179129, -0.0184203, -0.0179145, -0.0005309, 0.0005074

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000721
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202488, -0.0201303, -0.0202399, -0.0201201, -0.0001287, 0.0001096
1: -0.0191733, -0.0189426, -0.0191762, -0.0189592, -0.0002141, 0.0002336
2: -0.0191738, -0.0188793, -0.0191729, -0.0188883, -0.0002855, 0.0002936
3: -0.0182721, -0.0179763, -0.0182735, -0.0179654, -0.0003067, 0.0002973
4: -0.0184259, -0.0179472, -0.0184238, -0.0179356, -0.0004903, 0.0004766

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000721
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000754
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202488, -0.0201303, -0.0202394, -0.0201275, -0.0001213, 0.0001091
1: -0.0191733, -0.0189426, -0.0191736, -0.0189546, -0.0002187, 0.0002310
2: -0.0191738, -0.0188793, -0.0191706, -0.0188776, -0.0002962, 0.0002913
3: -0.0182721, -0.0179763, -0.0182700, -0.0179633, -0.0003088, 0.0002937
4: -0.0184259, -0.0179472, -0.0184203, -0.0179145, -0.0005115, 0.0004731

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000721
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000754
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202546, -0.0201285, -0.0202362, -0.0201202, -0.0001345, 0.0001077
1: -0.0191814, -0.0189257, -0.0191697, -0.0189688, -0.0002126, 0.0002439
2: -0.0191859, -0.0188575, -0.0191625, -0.0189050, -0.0002809, 0.0003050
3: -0.0182789, -0.0179524, -0.0182667, -0.0179784, -0.0003005, 0.0003143
4: -0.0184453, -0.0179129, -0.0184068, -0.0179647, -0.0004806, 0.0004939

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000724
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202546, -0.0201285, -0.0202319, -0.0201282, -0.0001265, 0.0001035
1: -0.0191814, -0.0189257, -0.0191662, -0.0189758, -0.0002056, 0.0002405
2: -0.0191859, -0.0188575, -0.0191586, -0.0189027, -0.0002831, 0.0003011
3: -0.0182789, -0.0179524, -0.0182628, -0.0179878, -0.0002910, 0.0003104
4: -0.0184453, -0.0179129, -0.0184005, -0.0179535, -0.0004919, 0.0004876

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000724
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202488, -0.0201303, -0.0202362, -0.0201202, -0.0001287, 0.0001059
1: -0.0191733, -0.0189426, -0.0191697, -0.0189688, -0.0002044, 0.0002271
2: -0.0191738, -0.0188793, -0.0191625, -0.0189050, -0.0002688, 0.0002832
3: -0.0182721, -0.0179763, -0.0182667, -0.0179784, -0.0002936, 0.0002904
4: -0.0184259, -0.0179472, -0.0184068, -0.0179647, -0.0004612, 0.0004596

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000724
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000759
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202488, -0.0201303, -0.0202319, -0.0201282, -0.0001207, 0.0001016
1: -0.0191733, -0.0189426, -0.0191662, -0.0189758, -0.0001975, 0.0002237
2: -0.0191738, -0.0188793, -0.0191586, -0.0189027, -0.0002711, 0.0002793
3: -0.0182721, -0.0179763, -0.0182628, -0.0179878, -0.0002842, 0.0002865
4: -0.0184259, -0.0179472, -0.0184005, -0.0179535, -0.0004725, 0.0004533

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000724
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000759
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202462, -0.0201283, -0.0202563, -0.0201201, -0.0001261, 0.0001280
1: -0.0191755, -0.0189421, -0.0191861, -0.0189257, -0.0002498, 0.0002440
2: -0.0191731, -0.0188655, -0.0191913, -0.0188647, -0.0003084, 0.0003258
3: -0.0182722, -0.0179506, -0.0182848, -0.0179524, -0.0003198, 0.0003341
4: -0.0184238, -0.0179001, -0.0184537, -0.0179286, -0.0004952, 0.0005536

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000737
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000737
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202405, -0.0201287, -0.0202563, -0.0201201, -0.0001204, 0.0001276
1: -0.0191687, -0.0189593, -0.0191861, -0.0189257, -0.0002430, 0.0002268
2: -0.0191614, -0.0188869, -0.0191913, -0.0188647, -0.0002968, 0.0003044
3: -0.0182653, -0.0179696, -0.0182848, -0.0179524, -0.0003129, 0.0003152
4: -0.0184047, -0.0179342, -0.0184537, -0.0179286, -0.0004761, 0.0005195

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000738
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000738
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202462, -0.0201283, -0.0202528, -0.0201227, -0.0001235, 0.0001245
1: -0.0191755, -0.0189421, -0.0191780, -0.0189334, -0.0002421, 0.0002359
2: -0.0191731, -0.0188655, -0.0191794, -0.0188810, -0.0002920, 0.0003138
3: -0.0182722, -0.0179506, -0.0182772, -0.0179695, -0.0003027, 0.0003266
4: -0.0184238, -0.0179001, -0.0184346, -0.0179542, -0.0004696, 0.0005345

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202405, -0.0201287, -0.0202528, -0.0201227, -0.0001178, 0.0001241
1: -0.0191687, -0.0189593, -0.0191780, -0.0189334, -0.0002352, 0.0002187
2: -0.0191614, -0.0188869, -0.0191794, -0.0188810, -0.0002804, 0.0002924
3: -0.0182653, -0.0179696, -0.0182772, -0.0179695, -0.0002958, 0.0003077
4: -0.0184047, -0.0179342, -0.0184346, -0.0179542, -0.0004505, 0.0005005

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202462, -0.0201283, -0.0202546, -0.0201285, -0.0001178, 0.0001263
1: -0.0191755, -0.0189421, -0.0191814, -0.0189257, -0.0002498, 0.0002393
2: -0.0191731, -0.0188655, -0.0191859, -0.0188575, -0.0003156, 0.0003203
3: -0.0182722, -0.0179506, -0.0182789, -0.0179524, -0.0003198, 0.0003283
4: -0.0184238, -0.0179001, -0.0184453, -0.0179129, -0.0005109, 0.0005453

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000718
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000718
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202462, -0.0201283, -0.0202488, -0.0201303, -0.0001159, 0.0001205
1: -0.0191755, -0.0189421, -0.0191733, -0.0189426, -0.0002329, 0.0002312
2: -0.0191731, -0.0188655, -0.0191738, -0.0188793, -0.0002937, 0.0003083
3: -0.0182722, -0.0179506, -0.0182721, -0.0179763, -0.0002959, 0.0003215
4: -0.0184238, -0.0179001, -0.0184259, -0.0179472, -0.0004766, 0.0005259

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000718
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000722
time: 0.23 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202405, -0.0201287, -0.0202546, -0.0201285, -0.0001120, 0.0001259
1: -0.0191687, -0.0189593, -0.0191814, -0.0189257, -0.0002429, 0.0002221
2: -0.0191614, -0.0188869, -0.0191859, -0.0188575, -0.0003039, 0.0002989
3: -0.0182653, -0.0179696, -0.0182789, -0.0179524, -0.0003129, 0.0003093
4: -0.0184047, -0.0179342, -0.0184453, -0.0179129, -0.0004918, 0.0005112

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000719
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000719
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202405, -0.0201287, -0.0202488, -0.0201303, -0.0001102, 0.0001201
1: -0.0191687, -0.0189593, -0.0191733, -0.0189426, -0.0002261, 0.0002140
2: -0.0191614, -0.0188869, -0.0191738, -0.0188793, -0.0002821, 0.0002869
3: -0.0182653, -0.0179696, -0.0182721, -0.0179763, -0.0002890, 0.0003025
4: -0.0184047, -0.0179342, -0.0184259, -0.0179472, -0.0004575, 0.0004918

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000724
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000724
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202380, -0.0201210, -0.0001171, 0.0001169
1: -0.0191756, -0.0189626, -0.0191717, -0.0189639, -0.0002117, 0.0002091
2: -0.0191720, -0.0188924, -0.0191650, -0.0188997, -0.0002723, 0.0002726
3: -0.0182728, -0.0179693, -0.0182697, -0.0179730, -0.0002998, 0.0003004
4: -0.0184224, -0.0179407, -0.0184101, -0.0179563, -0.0004660, 0.0004693

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000737
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202343, -0.0201291, -0.0001091, 0.0001132
1: -0.0191756, -0.0189626, -0.0191682, -0.0189696, -0.0002060, 0.0002056
2: -0.0191720, -0.0188924, -0.0191613, -0.0188954, -0.0002765, 0.0002689
3: -0.0182728, -0.0179693, -0.0182656, -0.0179802, -0.0002926, 0.0002963
4: -0.0184224, -0.0179407, -0.0184042, -0.0179424, -0.0004800, 0.0004635

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000737
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202380, -0.0201210, -0.0001172, 0.0001090
1: -0.0191733, -0.0189577, -0.0191717, -0.0189639, -0.0002095, 0.0002140
2: -0.0191702, -0.0188809, -0.0191650, -0.0188997, -0.0002706, 0.0002841
3: -0.0182697, -0.0179663, -0.0182697, -0.0179730, -0.0002967, 0.0003034
4: -0.0184198, -0.0179187, -0.0184101, -0.0179563, -0.0004635, 0.0004914

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000736
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000747
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202343, -0.0201291, -0.0001092, 0.0001053
1: -0.0191733, -0.0189577, -0.0191682, -0.0189696, -0.0002037, 0.0002105
2: -0.0191702, -0.0188809, -0.0191613, -0.0188954, -0.0002748, 0.0002804
3: -0.0182697, -0.0179663, -0.0182656, -0.0179802, -0.0002895, 0.0002993
4: -0.0184198, -0.0179187, -0.0184042, -0.0179424, -0.0004774, 0.0004855

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000736
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000747
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202348, -0.0201212, -0.0202380, -0.0201210, -0.0001138, 0.0001168
1: -0.0191690, -0.0189719, -0.0191717, -0.0189639, -0.0002051, 0.0001998
2: -0.0191616, -0.0189085, -0.0191650, -0.0188997, -0.0002620, 0.0002565
3: -0.0182660, -0.0179830, -0.0182697, -0.0179730, -0.0002930, 0.0002867
4: -0.0184055, -0.0179693, -0.0184101, -0.0179563, -0.0004492, 0.0004408

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000745
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000752
time: 0.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202309, -0.0201292, -0.0202380, -0.0201210, -0.0001098, 0.0001087
1: -0.0191656, -0.0189782, -0.0191717, -0.0189639, -0.0002018, 0.0001935
2: -0.0191579, -0.0189057, -0.0191650, -0.0188997, -0.0002582, 0.0002593
3: -0.0182621, -0.0179915, -0.0182697, -0.0179730, -0.0002891, 0.0002782
4: -0.0183995, -0.0179574, -0.0184101, -0.0179563, -0.0004432, 0.0004527

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000745
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000753
time: 0.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202348, -0.0201212, -0.0202343, -0.0201291, -0.0001057, 0.0001132
1: -0.0191690, -0.0189719, -0.0191682, -0.0189696, -0.0001994, 0.0001964
2: -0.0191616, -0.0189085, -0.0191613, -0.0188954, -0.0002662, 0.0002528
3: -0.0182660, -0.0179830, -0.0182656, -0.0179802, -0.0002858, 0.0002826
4: -0.0184055, -0.0179693, -0.0184042, -0.0179424, -0.0004632, 0.0004349

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000751
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202309, -0.0201292, -0.0202343, -0.0201291, -0.0001018, 0.0001051
1: -0.0191656, -0.0189782, -0.0191682, -0.0189696, -0.0001960, 0.0001900
2: -0.0191579, -0.0189057, -0.0191613, -0.0188954, -0.0002625, 0.0002557
3: -0.0182621, -0.0179915, -0.0182656, -0.0179802, -0.0002819, 0.0002741
4: -0.0183995, -0.0179574, -0.0184042, -0.0179424, -0.0004572, 0.0004469

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
time: 0.20 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 1.67 seconds
NS_A2_B2_A1_A1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000728, upper bound: 0.0000720
NS_A2_B2_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000728, upper bound: 0.0000749
NS_A2_B2_A1_A1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000728, upper bound: 0.0000720
NS_A2_B2_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000728, upper bound: 0.0000749
NS_A2_B2_A1_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000724
NS_A2_B2_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
NS_A2_B2_A1_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000724
NS_A2_B2_A1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000726, upper bound: 0.0000744
NS_A2_B2_A1_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000721
NS_A2_B2_A1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
NS_A2_B2_A1_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000721
NS_A2_B2_A1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000748
NS_A2_B2_A1_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000721
NS_A2_B2_A1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000754
NS_A2_B2_A1_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000721
NS_A2_B2_A1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000754
NS_A2_B2_A1_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000724
NS_A2_B2_A1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
NS_A2_B2_A1_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000724
NS_A2_B2_A1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000747
NS_A2_B2_A1_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000724
NS_A2_B2_A1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000759
NS_A2_B2_A1_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000724
NS_A2_B2_A1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000759
NS_A2_B2_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000737
NS_A2_B2_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000737
NS_A2_B2_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000738
NS_A2_B2_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000738
NS_A2_B2_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
NS_A2_B2_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
NS_A2_B2_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
NS_A2_B2_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
NS_A2_B2_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000718
NS_A2_B2_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000718
NS_A2_B2_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000718
NS_A2_B2_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000722
NS_A2_B2_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000719
NS_A2_B2_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000719
NS_A2_B2_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000724
NS_A2_B2_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000724
NS_A2_B2_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
NS_A2_B2_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000737
NS_A2_B2_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
NS_A2_B2_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000737
NS_A2_B2_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000736
NS_A2_B2_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000747
NS_A2_B2_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000736
NS_A2_B2_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000752, upper bound: 0.0000747
NS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000745
NS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000752
NS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000745
NS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000753
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000751
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.67
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736

## BFS NS instance: NS_A2_B2_A1_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202381, -0.0201211, -0.0001317, 0.0001154
1: -0.0191780, -0.0189334, -0.0191756, -0.0189626, -0.0002154, 0.0002422
2: -0.0191794, -0.0188810, -0.0191720, -0.0188924, -0.0002870, 0.0002909
3: -0.0182772, -0.0179695, -0.0182728, -0.0179693, -0.0003079, 0.0003033
4: -0.0184346, -0.0179542, -0.0184224, -0.0179407, -0.0004939, 0.0004681

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202382, -0.0201290, -0.0001238, 0.0001155
1: -0.0191780, -0.0189334, -0.0191733, -0.0189577, -0.0002203, 0.0002399
2: -0.0191794, -0.0188810, -0.0191702, -0.0188809, -0.0002985, 0.0002892
3: -0.0182772, -0.0179695, -0.0182697, -0.0179663, -0.0003109, 0.0003002
4: -0.0184346, -0.0179542, -0.0184198, -0.0179187, -0.0005159, 0.0004656

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202348, -0.0201212, -0.0001316, 0.0001121
1: -0.0191780, -0.0189334, -0.0191690, -0.0189719, -0.0002061, 0.0002356
2: -0.0191794, -0.0188810, -0.0191616, -0.0189085, -0.0002708, 0.0002806
3: -0.0182772, -0.0179695, -0.0182660, -0.0179830, -0.0002942, 0.0002965
4: -0.0184346, -0.0179542, -0.0184055, -0.0179693, -0.0004653, 0.0004513

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202528, -0.0201227, -0.0202309, -0.0201292, -0.0001236, 0.0001081
1: -0.0191780, -0.0189334, -0.0191656, -0.0189782, -0.0001998, 0.0002322
2: -0.0191794, -0.0188810, -0.0191579, -0.0189057, -0.0002737, 0.0002769
3: -0.0182772, -0.0179695, -0.0182621, -0.0179915, -0.0002857, 0.0002926
4: -0.0184346, -0.0179542, -0.0183995, -0.0179574, -0.0004773, 0.0004453

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202546, -0.0201285, -0.0202381, -0.0201211, -0.0001336, 0.0001097
1: -0.0191814, -0.0189257, -0.0191756, -0.0189626, -0.0002188, 0.0002498
2: -0.0191859, -0.0188575, -0.0191720, -0.0188924, -0.0002935, 0.0003145
3: -0.0182789, -0.0179524, -0.0182728, -0.0179693, -0.0003096, 0.0003204
4: -0.0184453, -0.0179129, -0.0184224, -0.0179407, -0.0005046, 0.0005094

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202546, -0.0201285, -0.0202382, -0.0201290, -0.0001257, 0.0001098
1: -0.0191814, -0.0189257, -0.0191733, -0.0189577, -0.0002237, 0.0002476
2: -0.0191859, -0.0188575, -0.0191702, -0.0188809, -0.0003050, 0.0003127
3: -0.0182789, -0.0179524, -0.0182697, -0.0179663, -0.0003125, 0.0003173
4: -0.0184453, -0.0179129, -0.0184198, -0.0179187, -0.0005266, 0.0005069

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202488, -0.0201303, -0.0202381, -0.0201211, -0.0001277, 0.0001078
1: -0.0191733, -0.0189426, -0.0191756, -0.0189626, -0.0002107, 0.0002330
2: -0.0191738, -0.0188793, -0.0191720, -0.0188924, -0.0002814, 0.0002926
3: -0.0182721, -0.0179763, -0.0182728, -0.0179693, -0.0003028, 0.0002965
4: -0.0184259, -0.0179472, -0.0184224, -0.0179407, -0.0004852, 0.0004751

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202488, -0.0201303, -0.0202382, -0.0201290, -0.0001199, 0.0001079
1: -0.0191733, -0.0189426, -0.0191733, -0.0189577, -0.0002156, 0.0002308
2: -0.0191738, -0.0188793, -0.0191702, -0.0188809, -0.0002929, 0.0002909
3: -0.0182721, -0.0179763, -0.0182697, -0.0179663, -0.0003057, 0.0002934
4: -0.0184259, -0.0179472, -0.0184198, -0.0179187, -0.0005072, 0.0004725

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202546, -0.0201285, -0.0202348, -0.0201212, -0.0001335, 0.0001063
1: -0.0191814, -0.0189257, -0.0191690, -0.0189719, -0.0002095, 0.0002433
2: -0.0191859, -0.0188575, -0.0191616, -0.0189085, -0.0002773, 0.0003041
3: -0.0182789, -0.0179524, -0.0182660, -0.0179830, -0.0002959, 0.0003135
4: -0.0184453, -0.0179129, -0.0184055, -0.0179693, -0.0004760, 0.0004926

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202546, -0.0201285, -0.0202309, -0.0201292, -0.0001254, 0.0001024
1: -0.0191814, -0.0189257, -0.0191656, -0.0189782, -0.0002032, 0.0002399
2: -0.0191859, -0.0188575, -0.0191579, -0.0189057, -0.0002802, 0.0003004
3: -0.0182789, -0.0179524, -0.0182621, -0.0179915, -0.0002874, 0.0003097
4: -0.0184453, -0.0179129, -0.0183995, -0.0179574, -0.0004880, 0.0004866

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202488, -0.0201303, -0.0202348, -0.0201212, -0.0001277, 0.0001044
1: -0.0191733, -0.0189426, -0.0191690, -0.0189719, -0.0002014, 0.0002264
2: -0.0191738, -0.0188793, -0.0191616, -0.0189085, -0.0002653, 0.0002823
3: -0.0182721, -0.0179763, -0.0182660, -0.0179830, -0.0002891, 0.0002897
4: -0.0184259, -0.0179472, -0.0184055, -0.0179693, -0.0004566, 0.0004583

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A1_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202488, -0.0201303, -0.0202309, -0.0201292, -0.0001196, 0.0001005
1: -0.0191733, -0.0189426, -0.0191656, -0.0189782, -0.0001951, 0.0002231
2: -0.0191738, -0.0188793, -0.0191579, -0.0189057, -0.0002682, 0.0002786
3: -0.0182721, -0.0179763, -0.0182621, -0.0179915, -0.0002806, 0.0002858
4: -0.0184259, -0.0179472, -0.0183995, -0.0179574, -0.0004686, 0.0004523

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202563, -0.0201201, -0.0001180, 0.0001352
1: -0.0191756, -0.0189626, -0.0191861, -0.0189257, -0.0002499, 0.0002235
2: -0.0191720, -0.0188924, -0.0191913, -0.0188647, -0.0003073, 0.0002989
3: -0.0182728, -0.0179693, -0.0182848, -0.0179524, -0.0003204, 0.0003155
4: -0.0184224, -0.0179407, -0.0184537, -0.0179286, -0.0004937, 0.0005130

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202563, -0.0201201, -0.0001181, 0.0001273
1: -0.0191733, -0.0189577, -0.0191861, -0.0189257, -0.0002477, 0.0002284
2: -0.0191702, -0.0188809, -0.0191913, -0.0188647, -0.0003056, 0.0003104
3: -0.0182697, -0.0179663, -0.0182848, -0.0179524, -0.0003173, 0.0003184
4: -0.0184198, -0.0179187, -0.0184537, -0.0179286, -0.0004911, 0.0005350

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0202348, -0.0201212, -0.0202563, -0.0201201, -0.0001146, 0.0001352
1: -0.0191690, -0.0189719, -0.0191861, -0.0189257, -0.0002433, 0.0002142
2: -0.0191616, -0.0189085, -0.0191913, -0.0188647, -0.0002969, 0.0002828
3: -0.0182660, -0.0179830, -0.0182848, -0.0179524, -0.0003135, 0.0003018
4: -0.0184055, -0.0179693, -0.0184537, -0.0179286, -0.0004769, 0.0004844

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0202309, -0.0201292, -0.0202563, -0.0201201, -0.0001107, 0.0001271
1: -0.0191656, -0.0189782, -0.0191861, -0.0189257, -0.0002400, 0.0002079
2: -0.0191579, -0.0189057, -0.0191913, -0.0188647, -0.0002932, 0.0002857
3: -0.0182621, -0.0179915, -0.0182848, -0.0179524, -0.0003097, 0.0002933
4: -0.0183995, -0.0179574, -0.0184537, -0.0179286, -0.0004709, 0.0004963

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202528, -0.0201227, -0.0001154, 0.0001317
1: -0.0191756, -0.0189626, -0.0191780, -0.0189334, -0.0002422, 0.0002154
2: -0.0191720, -0.0188924, -0.0191794, -0.0188810, -0.0002909, 0.0002870
3: -0.0182728, -0.0179693, -0.0182772, -0.0179695, -0.0003033, 0.0003079
4: -0.0184224, -0.0179407, -0.0184346, -0.0179542, -0.0004681, 0.0004939

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202528, -0.0201227, -0.0001155, 0.0001238
1: -0.0191733, -0.0189577, -0.0191780, -0.0189334, -0.0002399, 0.0002203
2: -0.0191702, -0.0188809, -0.0191794, -0.0188810, -0.0002892, 0.0002985
3: -0.0182697, -0.0179663, -0.0182772, -0.0179695, -0.0003002, 0.0003109
4: -0.0184198, -0.0179187, -0.0184346, -0.0179542, -0.0004656, 0.0005159

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0202348, -0.0201212, -0.0202528, -0.0201227, -0.0001121, 0.0001316
1: -0.0191690, -0.0189719, -0.0191780, -0.0189334, -0.0002356, 0.0002061
2: -0.0191616, -0.0189085, -0.0191794, -0.0188810, -0.0002806, 0.0002708
3: -0.0182660, -0.0179830, -0.0182772, -0.0179695, -0.0002965, 0.0002942
4: -0.0184055, -0.0179693, -0.0184346, -0.0179542, -0.0004513, 0.0004653

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0202309, -0.0201292, -0.0202528, -0.0201227, -0.0001081, 0.0001236
1: -0.0191656, -0.0189782, -0.0191780, -0.0189334, -0.0002322, 0.0001998
2: -0.0191579, -0.0189057, -0.0191794, -0.0188810, -0.0002769, 0.0002737
3: -0.0182621, -0.0179915, -0.0182772, -0.0179695, -0.0002926, 0.0002857
4: -0.0183995, -0.0179574, -0.0184346, -0.0179542, -0.0004453, 0.0004773

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202546, -0.0201285, -0.0001097, 0.0001336
1: -0.0191756, -0.0189626, -0.0191814, -0.0189257, -0.0002498, 0.0002188
2: -0.0191720, -0.0188924, -0.0191859, -0.0188575, -0.0003145, 0.0002935
3: -0.0182728, -0.0179693, -0.0182789, -0.0179524, -0.0003204, 0.0003096
4: -0.0184224, -0.0179407, -0.0184453, -0.0179129, -0.0005094, 0.0005046

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202546, -0.0201285, -0.0001098, 0.0001257
1: -0.0191733, -0.0189577, -0.0191814, -0.0189257, -0.0002476, 0.0002237
2: -0.0191702, -0.0188809, -0.0191859, -0.0188575, -0.0003127, 0.0003050
3: -0.0182697, -0.0179663, -0.0182789, -0.0179524, -0.0003173, 0.0003125
4: -0.0184198, -0.0179187, -0.0184453, -0.0179129, -0.0005069, 0.0005266

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202488, -0.0201303, -0.0001078, 0.0001277
1: -0.0191756, -0.0189626, -0.0191733, -0.0189426, -0.0002330, 0.0002107
2: -0.0191720, -0.0188924, -0.0191738, -0.0188793, -0.0002926, 0.0002814
3: -0.0182728, -0.0179693, -0.0182721, -0.0179763, -0.0002965, 0.0003028
4: -0.0184224, -0.0179407, -0.0184259, -0.0179472, -0.0004751, 0.0004852

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202488, -0.0201303, -0.0001079, 0.0001199
1: -0.0191733, -0.0189577, -0.0191733, -0.0189426, -0.0002308, 0.0002156
2: -0.0191702, -0.0188809, -0.0191738, -0.0188793, -0.0002909, 0.0002929
3: -0.0182697, -0.0179663, -0.0182721, -0.0179763, -0.0002934, 0.0003057
4: -0.0184198, -0.0179187, -0.0184259, -0.0179472, -0.0004725, 0.0005072

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202348, -0.0201212, -0.0202546, -0.0201285, -0.0001063, 0.0001335
1: -0.0191690, -0.0189719, -0.0191814, -0.0189257, -0.0002433, 0.0002095
2: -0.0191616, -0.0189085, -0.0191859, -0.0188575, -0.0003041, 0.0002773
3: -0.0182660, -0.0179830, -0.0182789, -0.0179524, -0.0003135, 0.0002959
4: -0.0184055, -0.0179693, -0.0184453, -0.0179129, -0.0004926, 0.0004760

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202309, -0.0201292, -0.0202546, -0.0201285, -0.0001024, 0.0001254
1: -0.0191656, -0.0189782, -0.0191814, -0.0189257, -0.0002399, 0.0002032
2: -0.0191579, -0.0189057, -0.0191859, -0.0188575, -0.0003004, 0.0002802
3: -0.0182621, -0.0179915, -0.0182789, -0.0179524, -0.0003097, 0.0002874
4: -0.0183995, -0.0179574, -0.0184453, -0.0179129, -0.0004866, 0.0004880

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202348, -0.0201212, -0.0202488, -0.0201303, -0.0001044, 0.0001277
1: -0.0191690, -0.0189719, -0.0191733, -0.0189426, -0.0002264, 0.0002014
2: -0.0191616, -0.0189085, -0.0191738, -0.0188793, -0.0002823, 0.0002653
3: -0.0182660, -0.0179830, -0.0182721, -0.0179763, -0.0002897, 0.0002891
4: -0.0184055, -0.0179693, -0.0184259, -0.0179472, -0.0004583, 0.0004566

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202309, -0.0201292, -0.0202488, -0.0201303, -0.0001005, 0.0001196
1: -0.0191656, -0.0189782, -0.0191733, -0.0189426, -0.0002231, 0.0001951
2: -0.0191579, -0.0189057, -0.0191738, -0.0188793, -0.0002786, 0.0002682
3: -0.0182621, -0.0179915, -0.0182721, -0.0179763, -0.0002858, 0.0002806
4: -0.0183995, -0.0179574, -0.0184259, -0.0179472, -0.0004523, 0.0004686

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202381, -0.0201211, -0.0001170, 0.0001170
1: -0.0191756, -0.0189626, -0.0191756, -0.0189626, -0.0002130, 0.0002130
2: -0.0191720, -0.0188924, -0.0191720, -0.0188924, -0.0002796, 0.0002796
3: -0.0182728, -0.0179693, -0.0182728, -0.0179693, -0.0003035, 0.0003035
4: -0.0184224, -0.0179407, -0.0184224, -0.0179407, -0.0004816, 0.0004816

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202348, -0.0201212, -0.0001170, 0.0001137
1: -0.0191756, -0.0189626, -0.0191690, -0.0189719, -0.0002037, 0.0002064
2: -0.0191720, -0.0188924, -0.0191616, -0.0189085, -0.0002634, 0.0002692
3: -0.0182728, -0.0179693, -0.0182660, -0.0179830, -0.0002898, 0.0002967
4: -0.0184224, -0.0179407, -0.0184055, -0.0179693, -0.0004531, 0.0004648

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202380, -0.0201290, -0.0001091, 0.0001169
1: -0.0191756, -0.0189626, -0.0191733, -0.0189582, -0.0002174, 0.0002107
2: -0.0191720, -0.0188924, -0.0191701, -0.0188815, -0.0002904, 0.0002777
3: -0.0182728, -0.0179693, -0.0182697, -0.0179670, -0.0003059, 0.0003004
4: -0.0184224, -0.0179407, -0.0184196, -0.0179195, -0.0005029, 0.0004789

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202381, -0.0201211, -0.0202309, -0.0201292, -0.0001089, 0.0001098
1: -0.0191756, -0.0189626, -0.0191656, -0.0189782, -0.0001974, 0.0002030
2: -0.0191720, -0.0188924, -0.0191579, -0.0189057, -0.0002663, 0.0002655
3: -0.0182728, -0.0179693, -0.0182621, -0.0179915, -0.0002813, 0.0002928
4: -0.0184224, -0.0179407, -0.0183995, -0.0179574, -0.0004650, 0.0004588

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202381, -0.0201211, -0.0001171, 0.0001091
1: -0.0191733, -0.0189577, -0.0191756, -0.0189626, -0.0002107, 0.0002179
2: -0.0191702, -0.0188809, -0.0191720, -0.0188924, -0.0002778, 0.0002911
3: -0.0182697, -0.0179663, -0.0182728, -0.0179693, -0.0003004, 0.0003065
4: -0.0184198, -0.0179187, -0.0184224, -0.0179407, -0.0004790, 0.0005037

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202348, -0.0201212, -0.0001171, 0.0001058
1: -0.0191733, -0.0189577, -0.0191690, -0.0189719, -0.0002015, 0.0002113
2: -0.0191702, -0.0188809, -0.0191616, -0.0189085, -0.0002617, 0.0002807
3: -0.0182697, -0.0179663, -0.0182660, -0.0179830, -0.0002867, 0.0002996
4: -0.0184198, -0.0179187, -0.0184055, -0.0179693, -0.0004505, 0.0004868

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202380, -0.0201290, -0.0001092, 0.0001090
1: -0.0191733, -0.0189577, -0.0191733, -0.0189582, -0.0002151, 0.0002156
2: -0.0191702, -0.0188809, -0.0191701, -0.0188815, -0.0002887, 0.0002893
3: -0.0182697, -0.0179663, -0.0182697, -0.0179670, -0.0003027, 0.0003033
4: -0.0184198, -0.0179187, -0.0184196, -0.0179195, -0.0005003, 0.0005009

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0202382, -0.0201290, -0.0202309, -0.0201292, -0.0001090, 0.0001019
1: -0.0191733, -0.0189577, -0.0191656, -0.0189782, -0.0001952, 0.0002079
2: -0.0191702, -0.0188809, -0.0191579, -0.0189057, -0.0002646, 0.0002770
3: -0.0182697, -0.0179663, -0.0182621, -0.0179915, -0.0002782, 0.0002958
4: -0.0184198, -0.0179187, -0.0183995, -0.0179574, -0.0004624, 0.0004808

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202348, -0.0201212, -0.0202381, -0.0201211, -0.0001137, 0.0001170
1: -0.0191690, -0.0189719, -0.0191756, -0.0189626, -0.0002064, 0.0002037
2: -0.0191616, -0.0189085, -0.0191720, -0.0188924, -0.0002692, 0.0002634
3: -0.0182660, -0.0179830, -0.0182728, -0.0179693, -0.0002967, 0.0002898
4: -0.0184055, -0.0179693, -0.0184224, -0.0179407, -0.0004648, 0.0004531

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202348, -0.0201212, -0.0202348, -0.0201212, -0.0001136, 0.0001136
1: -0.0191690, -0.0189719, -0.0191690, -0.0189719, -0.0001971, 0.0001971
2: -0.0191616, -0.0189085, -0.0191616, -0.0189085, -0.0002531, 0.0002531
3: -0.0182660, -0.0179830, -0.0182660, -0.0179830, -0.0002829, 0.0002829
4: -0.0184055, -0.0179693, -0.0184055, -0.0179693, -0.0004362, 0.0004362

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202309, -0.0201292, -0.0202381, -0.0201211, -0.0001098, 0.0001089
1: -0.0191656, -0.0189782, -0.0191756, -0.0189626, -0.0002030, 0.0001974
2: -0.0191579, -0.0189057, -0.0191720, -0.0188924, -0.0002655, 0.0002663
3: -0.0182621, -0.0179915, -0.0182728, -0.0179693, -0.0002928, 0.0002813
4: -0.0183995, -0.0179574, -0.0184224, -0.0179407, -0.0004588, 0.0004650

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202309, -0.0201292, -0.0202348, -0.0201212, -0.0001097, 0.0001055
1: -0.0191656, -0.0189782, -0.0191690, -0.0189719, -0.0001938, 0.0001908
2: -0.0191579, -0.0189057, -0.0191616, -0.0189085, -0.0002494, 0.0002560
3: -0.0182621, -0.0179915, -0.0182660, -0.0179830, -0.0002791, 0.0002744
4: -0.0183995, -0.0179574, -0.0184055, -0.0179693, -0.0004302, 0.0004482

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202348, -0.0201212, -0.0202380, -0.0201290, -0.0001058, 0.0001168
1: -0.0191690, -0.0189719, -0.0191733, -0.0189582, -0.0002108, 0.0002014
2: -0.0191616, -0.0189085, -0.0191701, -0.0188815, -0.0002801, 0.0002616
3: -0.0182660, -0.0179830, -0.0182697, -0.0179670, -0.0002990, 0.0002867
4: -0.0184055, -0.0179693, -0.0184196, -0.0179195, -0.0004861, 0.0004503

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202348, -0.0201212, -0.0202309, -0.0201292, -0.0001055, 0.0001097
1: -0.0191690, -0.0189719, -0.0191656, -0.0189782, -0.0001908, 0.0001938
2: -0.0191616, -0.0189085, -0.0191579, -0.0189057, -0.0002560, 0.0002494
3: -0.0182660, -0.0179830, -0.0182621, -0.0179915, -0.0002744, 0.0002791
4: -0.0184055, -0.0179693, -0.0183995, -0.0179574, -0.0004482, 0.0004302

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202309, -0.0201292, -0.0202380, -0.0201290, -0.0001019, 0.0001088
1: -0.0191656, -0.0189782, -0.0191733, -0.0189582, -0.0002074, 0.0001951
2: -0.0191579, -0.0189057, -0.0191701, -0.0188815, -0.0002764, 0.0002645
3: -0.0182621, -0.0179915, -0.0182697, -0.0179670, -0.0002952, 0.0002782
4: -0.0183995, -0.0179574, -0.0184196, -0.0179195, -0.0004800, 0.0004623

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202309, -0.0201292, -0.0202309, -0.0201292, -0.0001016, 0.0001016
1: -0.0191656, -0.0189782, -0.0191656, -0.0189782, -0.0001875, 0.0001875
2: -0.0191579, -0.0189057, -0.0191579, -0.0189057, -0.0002522, 0.0002522
3: -0.0182621, -0.0179915, -0.0182621, -0.0179915, -0.0002706, 0.0002706
4: -0.0183995, -0.0179574, -0.0183995, -0.0179574, -0.0004421, 0.0004421

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.47 + 208.84 = 210.31 seconds
