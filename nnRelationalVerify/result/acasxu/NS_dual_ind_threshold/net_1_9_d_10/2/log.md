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
execution time: IAR + RelationalAnalysis = 1.19 + 0.54 = 1.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000773, upper bound: 0.0000773

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000769
time: 0.15 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000769
time: 0.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.41 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.41
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000769
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.41
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000769

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0202436, -0.0201194, -0.0202495, -0.0201211, -0.0001225, 0.0001302
1: -0.0191808, -0.0189508, -0.0191812, -0.0189370, -0.0002438, 0.0002304
2: -0.0191744, -0.0188870, -0.0191748, -0.0188630, -0.0003114, 0.0002879
3: -0.0182789, -0.0179569, -0.0182793, -0.0179404, -0.0003384, 0.0003224
4: -0.0184253, -0.0179371, -0.0184249, -0.0178994, -0.0005259, 0.0004878

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000768
time: 0.16 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000769
time: 0.17 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0202401, -0.0201225, -0.0202495, -0.0201211, -0.0001190, 0.0001270
1: -0.0191780, -0.0189561, -0.0191812, -0.0189370, -0.0002411, 0.0002251
2: -0.0191711, -0.0188803, -0.0191748, -0.0188630, -0.0003081, 0.0002945
3: -0.0182758, -0.0179615, -0.0182793, -0.0179404, -0.0003354, 0.0003178
4: -0.0184194, -0.0179215, -0.0184249, -0.0178994, -0.0005200, 0.0005034

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000768
time: 0.18 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000769
time: 0.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.54 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.54
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000768
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.54
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000769
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.54
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000768
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.54
Output dim: 0, lower bound: -0.0000769, upper bound: 0.0000769

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202436, -0.0201194, -0.0202436, -0.0201194, -0.0001242, 0.0001242
1: -0.0191808, -0.0189508, -0.0191808, -0.0189508, -0.0002300, 0.0002300
2: -0.0191744, -0.0188870, -0.0191744, -0.0188870, -0.0002874, 0.0002874
3: -0.0182789, -0.0179569, -0.0182789, -0.0179569, -0.0003220, 0.0003220
4: -0.0184253, -0.0179371, -0.0184253, -0.0179371, -0.0004881, 0.0004881

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000763
time: 0.16 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000767
time: 0.17 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202436, -0.0201194, -0.0202401, -0.0201225, -0.0001211, 0.0001208
1: -0.0191808, -0.0189508, -0.0191780, -0.0189561, -0.0002246, 0.0002273
2: -0.0191744, -0.0188870, -0.0191711, -0.0188803, -0.0002941, 0.0002841
3: -0.0182789, -0.0179569, -0.0182758, -0.0179615, -0.0003173, 0.0003190
4: -0.0184253, -0.0179371, -0.0184194, -0.0179215, -0.0005038, 0.0004823

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000763
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000768
time: 0.17 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202401, -0.0201225, -0.0202436, -0.0201194, -0.0001208, 0.0001211
1: -0.0191780, -0.0189561, -0.0191808, -0.0189508, -0.0002273, 0.0002246
2: -0.0191711, -0.0188803, -0.0191744, -0.0188870, -0.0002841, 0.0002941
3: -0.0182758, -0.0179615, -0.0182789, -0.0179569, -0.0003190, 0.0003173
4: -0.0184194, -0.0179215, -0.0184253, -0.0179371, -0.0004823, 0.0005038

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000762
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000768
time: 0.15 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202401, -0.0201225, -0.0202401, -0.0201225, -0.0001176, 0.0001176
1: -0.0191780, -0.0189561, -0.0191780, -0.0189561, -0.0002219, 0.0002219
2: -0.0191711, -0.0188803, -0.0191711, -0.0188803, -0.0002908, 0.0002908
3: -0.0182758, -0.0179615, -0.0182758, -0.0179615, -0.0003143, 0.0003143
4: -0.0184194, -0.0179215, -0.0184194, -0.0179215, -0.0004979, 0.0004979

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000762
time: 0.16 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000768
time: 0.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.48 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.48
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000763
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.48
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000767
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.48
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000763
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.48
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000768
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.48
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000762
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.48
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000768
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.48
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000762
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.48
Output dim: 0, lower bound: -0.0000768, upper bound: 0.0000768

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202608, -0.0201189, -0.0202436, -0.0201194, -0.0001414, 0.0001247
1: -0.0191911, -0.0189193, -0.0191808, -0.0189508, -0.0002403, 0.0002615
2: -0.0191928, -0.0188630, -0.0191744, -0.0188870, -0.0003058, 0.0003114
3: -0.0182933, -0.0179489, -0.0182789, -0.0179569, -0.0003364, 0.0003299
4: -0.0184546, -0.0179231, -0.0184253, -0.0179371, -0.0005174, 0.0005021

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000742
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000765
time: 0.15 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202423, -0.0201205, -0.0202436, -0.0201194, -0.0001229, 0.0001231
1: -0.0191800, -0.0189541, -0.0191808, -0.0189508, -0.0002292, 0.0002267
2: -0.0191734, -0.0188903, -0.0191744, -0.0188870, -0.0002864, 0.0002841
3: -0.0182780, -0.0179610, -0.0182789, -0.0179569, -0.0003212, 0.0003179
4: -0.0184238, -0.0179425, -0.0184253, -0.0179371, -0.0004867, 0.0004828

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000742
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000742
time: 0.16 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202608, -0.0201189, -0.0202401, -0.0201225, -0.0001383, 0.0001212
1: -0.0191911, -0.0189193, -0.0191780, -0.0189561, -0.0002350, 0.0002588
2: -0.0191928, -0.0188630, -0.0191711, -0.0188803, -0.0003125, 0.0003081
3: -0.0182933, -0.0179489, -0.0182758, -0.0179615, -0.0003318, 0.0003269
4: -0.0184546, -0.0179231, -0.0184194, -0.0179215, -0.0005331, 0.0004963

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000725
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000763
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202423, -0.0201205, -0.0202401, -0.0201225, -0.0001197, 0.0001196
1: -0.0191800, -0.0189541, -0.0191780, -0.0189561, -0.0002238, 0.0002240
2: -0.0191734, -0.0188903, -0.0191711, -0.0188803, -0.0002931, 0.0002809
3: -0.0182780, -0.0179610, -0.0182758, -0.0179615, -0.0003165, 0.0003148
4: -0.0184238, -0.0179425, -0.0184194, -0.0179215, -0.0005023, 0.0004769

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000725
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000725
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202561, -0.0201239, -0.0202436, -0.0201194, -0.0001367, 0.0001197
1: -0.0191866, -0.0189300, -0.0191808, -0.0189508, -0.0002359, 0.0002507
2: -0.0191865, -0.0188615, -0.0191744, -0.0188870, -0.0002995, 0.0003129
3: -0.0182885, -0.0179584, -0.0182789, -0.0179569, -0.0003316, 0.0003204
4: -0.0184444, -0.0179164, -0.0184253, -0.0179371, -0.0005073, 0.0005088

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000742
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000765
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202387, -0.0201245, -0.0202436, -0.0201194, -0.0001193, 0.0001192
1: -0.0191773, -0.0189598, -0.0191808, -0.0189508, -0.0002265, 0.0002210
2: -0.0191701, -0.0188841, -0.0191744, -0.0188870, -0.0002831, 0.0002902
3: -0.0182750, -0.0179664, -0.0182789, -0.0179569, -0.0003181, 0.0003125
4: -0.0184180, -0.0179272, -0.0184253, -0.0179371, -0.0004808, 0.0004981

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000742
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000742
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202561, -0.0201239, -0.0202401, -0.0201225, -0.0001336, 0.0001162
1: -0.0191866, -0.0189300, -0.0191780, -0.0189561, -0.0002305, 0.0002480
2: -0.0191865, -0.0188615, -0.0191711, -0.0188803, -0.0003062, 0.0003096
3: -0.0182885, -0.0179584, -0.0182758, -0.0179615, -0.0003270, 0.0003174
4: -0.0184444, -0.0179164, -0.0184194, -0.0179215, -0.0005230, 0.0005030

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000762
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202387, -0.0201245, -0.0202401, -0.0201225, -0.0001162, 0.0001157
1: -0.0191773, -0.0189598, -0.0191780, -0.0189561, -0.0002211, 0.0002183
2: -0.0191701, -0.0188841, -0.0191711, -0.0188803, -0.0002898, 0.0002870
3: -0.0182750, -0.0179664, -0.0182758, -0.0179615, -0.0003135, 0.0003094
4: -0.0184180, -0.0179272, -0.0184194, -0.0179215, -0.0004965, 0.0004922

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000725
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000725
time: 0.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.54 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000742
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000765
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000742
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000742
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000725
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000743, upper bound: 0.0000763
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000725
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000725
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000742
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000765
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000742
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000742
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000725
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000725, upper bound: 0.0000762
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000725
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.54
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000725

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202608, -0.0201189, -0.0202608, -0.0201189, -0.0001419, 0.0001419
1: -0.0191911, -0.0189193, -0.0191911, -0.0189193, -0.0002718, 0.0002718
2: -0.0191928, -0.0188630, -0.0191928, -0.0188630, -0.0003297, 0.0003297
3: -0.0182933, -0.0179489, -0.0182933, -0.0179489, -0.0003444, 0.0003444
4: -0.0184546, -0.0179231, -0.0184546, -0.0179231, -0.0005314, 0.0005314

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000739
time: 0.17 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000738
time: 0.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202608, -0.0201189, -0.0202423, -0.0201205, -0.0001403, 0.0001233
1: -0.0191911, -0.0189193, -0.0191800, -0.0189541, -0.0002371, 0.0002607
2: -0.0191928, -0.0188630, -0.0191734, -0.0188903, -0.0003025, 0.0003104
3: -0.0182933, -0.0179489, -0.0182780, -0.0179610, -0.0003323, 0.0003291
4: -0.0184546, -0.0179231, -0.0184238, -0.0179425, -0.0005121, 0.0005007

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000764
time: 0.17 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000764
time: 0.15 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202423, -0.0201205, -0.0202608, -0.0201189, -0.0001233, 0.0001403
1: -0.0191800, -0.0189541, -0.0191911, -0.0189193, -0.0002607, 0.0002371
2: -0.0191734, -0.0188903, -0.0191928, -0.0188630, -0.0003104, 0.0003025
3: -0.0182780, -0.0179610, -0.0182933, -0.0179489, -0.0003291, 0.0003323
4: -0.0184238, -0.0179425, -0.0184546, -0.0179231, -0.0005007, 0.0005121

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000736
time: 0.17 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000737
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202423, -0.0201205, -0.0202423, -0.0201205, -0.0001218, 0.0001218
1: -0.0191800, -0.0189541, -0.0191800, -0.0189541, -0.0002259, 0.0002259
2: -0.0191734, -0.0188903, -0.0191734, -0.0188903, -0.0002831, 0.0002831
3: -0.0182780, -0.0179610, -0.0182780, -0.0179610, -0.0003170, 0.0003170
4: -0.0184238, -0.0179425, -0.0184238, -0.0179425, -0.0004813, 0.0004813

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000752
time: 0.17 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000737
time: 0.17 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202608, -0.0201189, -0.0202561, -0.0201239, -0.0001369, 0.0001372
1: -0.0191911, -0.0189193, -0.0191866, -0.0189300, -0.0002611, 0.0002674
2: -0.0191928, -0.0188630, -0.0191865, -0.0188615, -0.0003313, 0.0003235
3: -0.0182933, -0.0179489, -0.0182885, -0.0179584, -0.0003349, 0.0003395
4: -0.0184546, -0.0179231, -0.0184444, -0.0179164, -0.0005381, 0.0005213

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000726
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000726
time: 0.16 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202608, -0.0201189, -0.0202387, -0.0201245, -0.0001363, 0.0001198
1: -0.0191911, -0.0189193, -0.0191773, -0.0189598, -0.0002313, 0.0002580
2: -0.0191928, -0.0188630, -0.0191701, -0.0188841, -0.0003086, 0.0003071
3: -0.0182933, -0.0179489, -0.0182750, -0.0179664, -0.0003269, 0.0003260
4: -0.0184546, -0.0179231, -0.0184180, -0.0179272, -0.0005274, 0.0004948

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000762
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000762
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202423, -0.0201205, -0.0202561, -0.0201239, -0.0001184, 0.0001356
1: -0.0191800, -0.0189541, -0.0191866, -0.0189300, -0.0002499, 0.0002326
2: -0.0191734, -0.0188903, -0.0191865, -0.0188615, -0.0003119, 0.0002962
3: -0.0182780, -0.0179610, -0.0182885, -0.0179584, -0.0003196, 0.0003275
4: -0.0184238, -0.0179425, -0.0184444, -0.0179164, -0.0005074, 0.0005019

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000723
time: 0.18 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000724
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202423, -0.0201205, -0.0202387, -0.0201245, -0.0001178, 0.0001182
1: -0.0191800, -0.0189541, -0.0191773, -0.0189598, -0.0002202, 0.0002232
2: -0.0191734, -0.0188903, -0.0191701, -0.0188841, -0.0002892, 0.0002798
3: -0.0182780, -0.0179610, -0.0182750, -0.0179664, -0.0003116, 0.0003140
4: -0.0184238, -0.0179425, -0.0184180, -0.0179272, -0.0004966, 0.0004755

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000751
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000724
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202561, -0.0201239, -0.0202608, -0.0201189, -0.0001372, 0.0001369
1: -0.0191866, -0.0189300, -0.0191911, -0.0189193, -0.0002674, 0.0002611
2: -0.0191865, -0.0188615, -0.0191928, -0.0188630, -0.0003235, 0.0003313
3: -0.0182885, -0.0179584, -0.0182933, -0.0179489, -0.0003395, 0.0003349
4: -0.0184444, -0.0179164, -0.0184546, -0.0179231, -0.0005213, 0.0005381

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000719, upper bound: 0.0000739
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000738
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202561, -0.0201239, -0.0202423, -0.0201205, -0.0001356, 0.0001184
1: -0.0191866, -0.0189300, -0.0191800, -0.0189541, -0.0002326, 0.0002499
2: -0.0191865, -0.0188615, -0.0191734, -0.0188903, -0.0002962, 0.0003119
3: -0.0182885, -0.0179584, -0.0182780, -0.0179610, -0.0003275, 0.0003196
4: -0.0184444, -0.0179164, -0.0184238, -0.0179425, -0.0005019, 0.0005074

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000719, upper bound: 0.0000756
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000764
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202387, -0.0201245, -0.0202608, -0.0201189, -0.0001198, 0.0001363
1: -0.0191773, -0.0189598, -0.0191911, -0.0189193, -0.0002580, 0.0002313
2: -0.0191701, -0.0188841, -0.0191928, -0.0188630, -0.0003071, 0.0003086
3: -0.0182750, -0.0179664, -0.0182933, -0.0179489, -0.0003260, 0.0003269
4: -0.0184180, -0.0179272, -0.0184546, -0.0179231, -0.0004948, 0.0005274

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000736
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202387, -0.0201245, -0.0202423, -0.0201205, -0.0001182, 0.0001178
1: -0.0191773, -0.0189598, -0.0191800, -0.0189541, -0.0002232, 0.0002202
2: -0.0191701, -0.0188841, -0.0191734, -0.0188903, -0.0002798, 0.0002892
3: -0.0182750, -0.0179664, -0.0182780, -0.0179610, -0.0003140, 0.0003116
4: -0.0184180, -0.0179272, -0.0184238, -0.0179425, -0.0004755, 0.0004966

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000752
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000753
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202561, -0.0201239, -0.0202387, -0.0201245, -0.0001316, 0.0001148
1: -0.0191866, -0.0189300, -0.0191773, -0.0189598, -0.0002269, 0.0002472
2: -0.0191865, -0.0188615, -0.0191701, -0.0188841, -0.0003024, 0.0003086
3: -0.0182885, -0.0179584, -0.0182750, -0.0179664, -0.0003221, 0.0003166
4: -0.0184444, -0.0179164, -0.0184180, -0.0179272, -0.0005173, 0.0005015

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000719, upper bound: 0.0000753
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000761
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202387, -0.0201245, -0.0202561, -0.0201239, -0.0001148, 0.0001316
1: -0.0191773, -0.0189598, -0.0191866, -0.0189300, -0.0002472, 0.0002269
2: -0.0191701, -0.0188841, -0.0191865, -0.0188615, -0.0003086, 0.0003024
3: -0.0182750, -0.0179664, -0.0182885, -0.0179584, -0.0003166, 0.0003221
4: -0.0184180, -0.0179272, -0.0184444, -0.0179164, -0.0005015, 0.0005173

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000723
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000724
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202387, -0.0201245, -0.0202387, -0.0201245, -0.0001142, 0.0001142
1: -0.0191773, -0.0189598, -0.0191773, -0.0189598, -0.0002175, 0.0002175
2: -0.0191701, -0.0188841, -0.0191701, -0.0188841, -0.0002860, 0.0002860
3: -0.0182750, -0.0179664, -0.0182750, -0.0179664, -0.0003086, 0.0003086
4: -0.0184180, -0.0179272, -0.0184180, -0.0179272, -0.0004908, 0.0004908

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000751
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000724
time: 0.19 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.85 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000739
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000738
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000764
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000764
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000736
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000737
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000752
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000737
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000726
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000726
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000742, upper bound: 0.0000762
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000762
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000723
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000724
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000751
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000765, upper bound: 0.0000724
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000719, upper bound: 0.0000739
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000738
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000719, upper bound: 0.0000756
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000764
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000736
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000736
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000752
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000753
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000719, upper bound: 0.0000753
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000724, upper bound: 0.0000761
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000723
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000724
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000751
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.85
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000724

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202624, -0.0201167, -0.0202423, -0.0201205, -0.0001419, 0.0001255
1: -0.0191982, -0.0189157, -0.0191800, -0.0189541, -0.0002442, 0.0002643
2: -0.0192029, -0.0188516, -0.0191734, -0.0188903, -0.0003126, 0.0003218
3: -0.0183009, -0.0179404, -0.0182780, -0.0179610, -0.0003399, 0.0003376
4: -0.0184696, -0.0179078, -0.0184238, -0.0179425, -0.0005271, 0.0005160

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000764
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000764
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202582, -0.0201194, -0.0202423, -0.0201205, -0.0001377, 0.0001228
1: -0.0191893, -0.0189249, -0.0191800, -0.0189541, -0.0002353, 0.0002551
2: -0.0191893, -0.0188699, -0.0191734, -0.0188903, -0.0002991, 0.0003035
3: -0.0182902, -0.0179572, -0.0182780, -0.0179610, -0.0003292, 0.0003208
4: -0.0184495, -0.0179340, -0.0184238, -0.0179425, -0.0005070, 0.0004898

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000764
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000764
time: 0.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202608, -0.0201189, -0.0001250, 0.0001426
1: -0.0191872, -0.0189515, -0.0191911, -0.0189193, -0.0002679, 0.0002396
2: -0.0191840, -0.0188793, -0.0191928, -0.0188630, -0.0003210, 0.0003135
3: -0.0182851, -0.0179569, -0.0182933, -0.0179489, -0.0003362, 0.0003364
4: -0.0184406, -0.0179232, -0.0184546, -0.0179231, -0.0005174, 0.0005314

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000736
time: 0.17 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000736
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202608, -0.0201189, -0.0001202, 0.0001400
1: -0.0191775, -0.0189624, -0.0191911, -0.0189193, -0.0002582, 0.0002287
2: -0.0191701, -0.0188988, -0.0191928, -0.0188630, -0.0003070, 0.0002940
3: -0.0182747, -0.0179704, -0.0182933, -0.0179489, -0.0003258, 0.0003229
4: -0.0184192, -0.0179553, -0.0184546, -0.0179231, -0.0004960, 0.0004992

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000737
time: 0.16 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000737
time: 0.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202423, -0.0201205, -0.0001234, 0.0001241
1: -0.0191872, -0.0189515, -0.0191800, -0.0189541, -0.0002331, 0.0002285
2: -0.0191840, -0.0188793, -0.0191734, -0.0188903, -0.0002938, 0.0002941
3: -0.0182851, -0.0179569, -0.0182780, -0.0179610, -0.0003242, 0.0003211
4: -0.0184406, -0.0179232, -0.0184238, -0.0179425, -0.0004981, 0.0005006

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000746
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000752
time: 0.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202423, -0.0201205, -0.0001186, 0.0001215
1: -0.0191775, -0.0189624, -0.0191800, -0.0189541, -0.0002234, 0.0002175
2: -0.0191701, -0.0188988, -0.0191734, -0.0188903, -0.0002798, 0.0002746
3: -0.0182747, -0.0179704, -0.0182780, -0.0179610, -0.0003137, 0.0003076
4: -0.0184192, -0.0179553, -0.0184238, -0.0179425, -0.0004767, 0.0004685

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000746
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000746
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202624, -0.0201167, -0.0202387, -0.0201245, -0.0001379, 0.0001220
1: -0.0191982, -0.0189157, -0.0191773, -0.0189598, -0.0002384, 0.0002616
2: -0.0192029, -0.0188516, -0.0191701, -0.0188841, -0.0003187, 0.0003185
3: -0.0183009, -0.0179404, -0.0182750, -0.0179664, -0.0003345, 0.0003345
4: -0.0184696, -0.0179078, -0.0184180, -0.0179272, -0.0005425, 0.0005102

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000756
time: 0.18 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000762
time: 0.17 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202582, -0.0201194, -0.0202387, -0.0201245, -0.0001338, 0.0001193
1: -0.0191893, -0.0189249, -0.0191773, -0.0189598, -0.0002295, 0.0002524
2: -0.0191893, -0.0188699, -0.0191701, -0.0188841, -0.0003052, 0.0003002
3: -0.0182902, -0.0179572, -0.0182750, -0.0179664, -0.0003238, 0.0003178
4: -0.0184495, -0.0179340, -0.0184180, -0.0179272, -0.0005223, 0.0004840

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000756
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000756
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202561, -0.0201239, -0.0001200, 0.0001379
1: -0.0191872, -0.0189515, -0.0191866, -0.0189300, -0.0002572, 0.0002351
2: -0.0191840, -0.0188793, -0.0191865, -0.0188615, -0.0003226, 0.0003072
3: -0.0182851, -0.0179569, -0.0182885, -0.0179584, -0.0003267, 0.0003316
4: -0.0184406, -0.0179232, -0.0184444, -0.0179164, -0.0005241, 0.0005212

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000718
time: 0.18 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000718
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202561, -0.0201239, -0.0001152, 0.0001353
1: -0.0191775, -0.0189624, -0.0191866, -0.0189300, -0.0002475, 0.0002242
2: -0.0191701, -0.0188988, -0.0191865, -0.0188615, -0.0003086, 0.0002877
3: -0.0182747, -0.0179704, -0.0182885, -0.0179584, -0.0003163, 0.0003181
4: -0.0184192, -0.0179553, -0.0184444, -0.0179164, -0.0005027, 0.0004891

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000719
time: 0.18 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000724
time: 0.17 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202387, -0.0201245, -0.0001194, 0.0001205
1: -0.0191872, -0.0189515, -0.0191773, -0.0189598, -0.0002274, 0.0002258
2: -0.0191840, -0.0188793, -0.0191701, -0.0188841, -0.0002999, 0.0002908
3: -0.0182851, -0.0179569, -0.0182750, -0.0179664, -0.0003187, 0.0003181
4: -0.0184406, -0.0179232, -0.0184180, -0.0179272, -0.0005134, 0.0004948

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000736
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000751
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202387, -0.0201245, -0.0001147, 0.0001179
1: -0.0191775, -0.0189624, -0.0191773, -0.0189598, -0.0002177, 0.0002148
2: -0.0191701, -0.0188988, -0.0191701, -0.0188841, -0.0002859, 0.0002713
3: -0.0182747, -0.0179704, -0.0182750, -0.0179664, -0.0003083, 0.0003046
4: -0.0184192, -0.0179553, -0.0184180, -0.0179272, -0.0004920, 0.0004626

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000737
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000752
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202577, -0.0201233, -0.0202423, -0.0201205, -0.0001372, 0.0001190
1: -0.0191930, -0.0189217, -0.0191800, -0.0189541, -0.0002390, 0.0002583
2: -0.0191953, -0.0188504, -0.0191734, -0.0188903, -0.0003051, 0.0003230
3: -0.0182952, -0.0179459, -0.0182780, -0.0179610, -0.0003342, 0.0003321
4: -0.0184582, -0.0179009, -0.0184238, -0.0179425, -0.0005157, 0.0005229

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000756
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000756
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202534, -0.0201245, -0.0202423, -0.0201205, -0.0001329, 0.0001177
1: -0.0191850, -0.0189358, -0.0191800, -0.0189541, -0.0002309, 0.0002441
2: -0.0191833, -0.0188694, -0.0191734, -0.0188903, -0.0002930, 0.0003040
3: -0.0182856, -0.0179656, -0.0182780, -0.0179610, -0.0003247, 0.0003125
4: -0.0184397, -0.0179284, -0.0184238, -0.0179425, -0.0004972, 0.0004954

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000764
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000765
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202608, -0.0201189, -0.0001225, 0.0001363
1: -0.0191837, -0.0189509, -0.0191911, -0.0189193, -0.0002644, 0.0002402
2: -0.0191792, -0.0188723, -0.0191928, -0.0188630, -0.0003162, 0.0003204
3: -0.0182814, -0.0179578, -0.0182933, -0.0179489, -0.0003325, 0.0003355
4: -0.0184330, -0.0179065, -0.0184546, -0.0179231, -0.0005098, 0.0005480

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202608, -0.0201189, -0.0001163, 0.0001356
1: -0.0191748, -0.0189689, -0.0191911, -0.0189193, -0.0002555, 0.0002222
2: -0.0191666, -0.0188941, -0.0191928, -0.0188630, -0.0003036, 0.0002986
3: -0.0182717, -0.0179773, -0.0182933, -0.0179489, -0.0003228, 0.0003160
4: -0.0184131, -0.0179416, -0.0184546, -0.0179231, -0.0004900, 0.0005130

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000736
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000736
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202423, -0.0201205, -0.0001209, 0.0001178
1: -0.0191837, -0.0189509, -0.0191800, -0.0189541, -0.0002296, 0.0002290
2: -0.0191792, -0.0188723, -0.0191734, -0.0188903, -0.0002889, 0.0003011
3: -0.0182814, -0.0179578, -0.0182780, -0.0179610, -0.0003204, 0.0003202
4: -0.0184330, -0.0179065, -0.0184238, -0.0179425, -0.0004905, 0.0005173

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000746
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000752
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202423, -0.0201205, -0.0001147, 0.0001171
1: -0.0191748, -0.0189689, -0.0191800, -0.0189541, -0.0002207, 0.0002110
2: -0.0191666, -0.0188941, -0.0191734, -0.0188903, -0.0002763, 0.0002793
3: -0.0182717, -0.0179773, -0.0182780, -0.0179610, -0.0003107, 0.0003007
4: -0.0184131, -0.0179416, -0.0184238, -0.0179425, -0.0004706, 0.0004822

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000746
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000753
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202577, -0.0201233, -0.0202387, -0.0201245, -0.0001333, 0.0001154
1: -0.0191930, -0.0189217, -0.0191773, -0.0189598, -0.0002332, 0.0002556
2: -0.0191953, -0.0188504, -0.0191701, -0.0188841, -0.0003112, 0.0003197
3: -0.0182952, -0.0179459, -0.0182750, -0.0179664, -0.0003288, 0.0003290
4: -0.0184582, -0.0179009, -0.0184180, -0.0179272, -0.0005310, 0.0005171

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000753
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000753
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202534, -0.0201245, -0.0202387, -0.0201245, -0.0001289, 0.0001142
1: -0.0191850, -0.0189358, -0.0191773, -0.0189598, -0.0002252, 0.0002414
2: -0.0191833, -0.0188694, -0.0191701, -0.0188841, -0.0002992, 0.0003007
3: -0.0182856, -0.0179656, -0.0182750, -0.0179664, -0.0003192, 0.0003094
4: -0.0184397, -0.0179284, -0.0184180, -0.0179272, -0.0005125, 0.0004895

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000756
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000761
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202561, -0.0201239, -0.0001175, 0.0001316
1: -0.0191837, -0.0189509, -0.0191866, -0.0189300, -0.0002536, 0.0002357
2: -0.0191792, -0.0188723, -0.0191865, -0.0188615, -0.0003177, 0.0003142
3: -0.0182814, -0.0179578, -0.0182885, -0.0179584, -0.0003230, 0.0003307
4: -0.0184330, -0.0179065, -0.0184444, -0.0179164, -0.0005165, 0.0005379

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000718
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000723
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202561, -0.0201239, -0.0001113, 0.0001309
1: -0.0191748, -0.0189689, -0.0191866, -0.0189300, -0.0002447, 0.0002177
2: -0.0191666, -0.0188941, -0.0191865, -0.0188615, -0.0003051, 0.0002924
3: -0.0182717, -0.0179773, -0.0182885, -0.0179584, -0.0003133, 0.0003112
4: -0.0184131, -0.0179416, -0.0184444, -0.0179164, -0.0004967, 0.0005028

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000719
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000724
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202387, -0.0201245, -0.0001170, 0.0001142
1: -0.0191837, -0.0189509, -0.0191773, -0.0189598, -0.0002239, 0.0002264
2: -0.0191792, -0.0188723, -0.0191701, -0.0188841, -0.0002951, 0.0002978
3: -0.0182814, -0.0179578, -0.0182750, -0.0179664, -0.0003150, 0.0003172
4: -0.0184330, -0.0179065, -0.0184180, -0.0179272, -0.0005058, 0.0005114

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000737
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000751
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202387, -0.0201245, -0.0001108, 0.0001135
1: -0.0191748, -0.0189689, -0.0191773, -0.0189598, -0.0002150, 0.0002083
2: -0.0191666, -0.0188941, -0.0191701, -0.0188841, -0.0002825, 0.0002760
3: -0.0182717, -0.0179773, -0.0182750, -0.0179664, -0.0003053, 0.0002977
4: -0.0184131, -0.0179416, -0.0184180, -0.0179272, -0.0004860, 0.0004764

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000757, upper bound: 0.0000736
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000757, upper bound: 0.0000736
time: 0.19 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.68 seconds
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000764
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000764
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000764
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000764
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000736
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000736
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000737
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000737
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000746
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000752
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000746
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000764, upper bound: 0.0000746
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000756
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000762
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000756
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000736, upper bound: 0.0000756
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000718
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000718
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000719
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000724
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000736
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000751
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000737
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000759, upper bound: 0.0000752
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000756
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000756
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000764
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000765
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000756, upper bound: 0.0000736
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000736
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000736
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000746
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000754, upper bound: 0.0000752
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000746
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000763, upper bound: 0.0000753
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000753
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000718, upper bound: 0.0000753
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000756
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000723, upper bound: 0.0000761
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000718
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000723
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000719
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000724
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000737
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000751
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000757, upper bound: 0.0000736
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.68
Output dim: 0, lower bound: -0.0000757, upper bound: 0.0000736

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202624, -0.0201167, -0.0202439, -0.0201182, -0.0001442, 0.0001272
1: -0.0191982, -0.0189157, -0.0191872, -0.0189515, -0.0002467, 0.0002715
2: -0.0192029, -0.0188516, -0.0191840, -0.0188793, -0.0003236, 0.0003325
3: -0.0183009, -0.0179404, -0.0182851, -0.0179569, -0.0003440, 0.0003447
4: -0.0184696, -0.0179078, -0.0184406, -0.0179232, -0.0005464, 0.0005328

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000761
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000762
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202624, -0.0201167, -0.0202391, -0.0201208, -0.0001416, 0.0001224
1: -0.0191982, -0.0189157, -0.0191775, -0.0189624, -0.0002358, 0.0002618
2: -0.0192029, -0.0188516, -0.0191701, -0.0188988, -0.0003041, 0.0003185
3: -0.0183009, -0.0179404, -0.0182747, -0.0179704, -0.0003305, 0.0003343
4: -0.0184696, -0.0179078, -0.0184192, -0.0179553, -0.0005143, 0.0005114

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000762
time: 0.17 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000737, upper bound: 0.0000762
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202582, -0.0201194, -0.0202439, -0.0201182, -0.0001401, 0.0001245
1: -0.0191893, -0.0189249, -0.0191872, -0.0189515, -0.0002378, 0.0002623
2: -0.0191893, -0.0188699, -0.0191840, -0.0188793, -0.0003100, 0.0003142
3: -0.0182902, -0.0179572, -0.0182851, -0.0179569, -0.0003333, 0.0003280
4: -0.0184495, -0.0179340, -0.0184406, -0.0179232, -0.0005263, 0.0005066

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202582, -0.0201194, -0.0202391, -0.0201208, -0.0001374, 0.0001197
1: -0.0191893, -0.0189249, -0.0191775, -0.0189624, -0.0002269, 0.0002526
2: -0.0191893, -0.0188699, -0.0191701, -0.0188988, -0.0002905, 0.0003002
3: -0.0182902, -0.0179572, -0.0182747, -0.0179704, -0.0003198, 0.0003175
4: -0.0184495, -0.0179340, -0.0184192, -0.0179553, -0.0004941, 0.0004852

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202624, -0.0201167, -0.0001272, 0.0001442
1: -0.0191872, -0.0189515, -0.0191982, -0.0189157, -0.0002715, 0.0002467
2: -0.0191840, -0.0188793, -0.0192029, -0.0188516, -0.0003325, 0.0003236
3: -0.0182851, -0.0179569, -0.0183009, -0.0179404, -0.0003447, 0.0003440
4: -0.0184406, -0.0179232, -0.0184696, -0.0179078, -0.0005328, 0.0005464

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000735
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000736
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202582, -0.0201194, -0.0001245, 0.0001401
1: -0.0191872, -0.0189515, -0.0191893, -0.0189249, -0.0002623, 0.0002378
2: -0.0191840, -0.0188793, -0.0191893, -0.0188699, -0.0003142, 0.0003100
3: -0.0182851, -0.0179569, -0.0182902, -0.0179572, -0.0003280, 0.0003333
4: -0.0184406, -0.0179232, -0.0184495, -0.0179340, -0.0005066, 0.0005263

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000735
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000736
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202624, -0.0201167, -0.0001224, 0.0001416
1: -0.0191775, -0.0189624, -0.0191982, -0.0189157, -0.0002618, 0.0002358
2: -0.0191701, -0.0188988, -0.0192029, -0.0188516, -0.0003185, 0.0003041
3: -0.0182747, -0.0179704, -0.0183009, -0.0179404, -0.0003343, 0.0003305
4: -0.0184192, -0.0179553, -0.0184696, -0.0179078, -0.0005114, 0.0005143

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000734
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000737
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202582, -0.0201194, -0.0001197, 0.0001374
1: -0.0191775, -0.0189624, -0.0191893, -0.0189249, -0.0002526, 0.0002269
2: -0.0191701, -0.0188988, -0.0191893, -0.0188699, -0.0003002, 0.0002905
3: -0.0182747, -0.0179704, -0.0182902, -0.0179572, -0.0003175, 0.0003198
4: -0.0184192, -0.0179553, -0.0184495, -0.0179340, -0.0004852, 0.0004941

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000749, upper bound: 0.0000734
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000736
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202439, -0.0201182, -0.0001257, 0.0001257
1: -0.0191872, -0.0189515, -0.0191872, -0.0189515, -0.0002357, 0.0002357
2: -0.0191840, -0.0188793, -0.0191840, -0.0188793, -0.0003047, 0.0003047
3: -0.0182851, -0.0179569, -0.0182851, -0.0179569, -0.0003282, 0.0003282
4: -0.0184406, -0.0179232, -0.0184406, -0.0179232, -0.0005174, 0.0005174

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000744
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000745
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202391, -0.0201208, -0.0001231, 0.0001209
1: -0.0191872, -0.0189515, -0.0191775, -0.0189624, -0.0002248, 0.0002260
2: -0.0191840, -0.0188793, -0.0191701, -0.0188988, -0.0002853, 0.0002908
3: -0.0182851, -0.0179569, -0.0182747, -0.0179704, -0.0003147, 0.0003178
4: -0.0184406, -0.0179232, -0.0184192, -0.0179553, -0.0004852, 0.0004960

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000753, upper bound: 0.0000744
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000761, upper bound: 0.0000752
time: 0.19 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202439, -0.0201182, -0.0001209, 0.0001231
1: -0.0191775, -0.0189624, -0.0191872, -0.0189515, -0.0002260, 0.0002248
2: -0.0191701, -0.0188988, -0.0191840, -0.0188793, -0.0002908, 0.0002853
3: -0.0182747, -0.0179704, -0.0182851, -0.0179569, -0.0003178, 0.0003147
4: -0.0184192, -0.0179553, -0.0184406, -0.0179232, -0.0004960, 0.0004852

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000743
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000745
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202391, -0.0201208, -0.0001183, 0.0001183
1: -0.0191775, -0.0189624, -0.0191775, -0.0189624, -0.0002150, 0.0002150
2: -0.0191701, -0.0188988, -0.0191701, -0.0188988, -0.0002713, 0.0002713
3: -0.0182747, -0.0179704, -0.0182747, -0.0179704, -0.0003043, 0.0003043
4: -0.0184192, -0.0179553, -0.0184192, -0.0179553, -0.0004638, 0.0004638

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000747, upper bound: 0.0000743
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000762, upper bound: 0.0000752
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202624, -0.0201167, -0.0202414, -0.0201245, -0.0001379, 0.0001247
1: -0.0191982, -0.0189157, -0.0191837, -0.0189509, -0.0002473, 0.0002680
2: -0.0192029, -0.0188516, -0.0191792, -0.0188723, -0.0003306, 0.0003276
3: -0.0183009, -0.0179404, -0.0182814, -0.0179578, -0.0003431, 0.0003410
4: -0.0184696, -0.0179078, -0.0184330, -0.0179065, -0.0005631, 0.0005252

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202624, -0.0201167, -0.0202352, -0.0201252, -0.0001372, 0.0001185
1: -0.0191982, -0.0189157, -0.0191748, -0.0189689, -0.0002293, 0.0002591
2: -0.0192029, -0.0188516, -0.0191666, -0.0188941, -0.0003087, 0.0003150
3: -0.0183009, -0.0179404, -0.0182717, -0.0179773, -0.0003236, 0.0003313
4: -0.0184696, -0.0179078, -0.0184131, -0.0179416, -0.0005281, 0.0005054

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202582, -0.0201194, -0.0202414, -0.0201245, -0.0001337, 0.0001220
1: -0.0191893, -0.0189249, -0.0191837, -0.0189509, -0.0002384, 0.0002588
2: -0.0191893, -0.0188699, -0.0191792, -0.0188723, -0.0003170, 0.0003093
3: -0.0182902, -0.0179572, -0.0182814, -0.0179578, -0.0003324, 0.0003243
4: -0.0184495, -0.0179340, -0.0184330, -0.0179065, -0.0005429, 0.0004989

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202582, -0.0201194, -0.0202352, -0.0201252, -0.0001330, 0.0001158
1: -0.0191893, -0.0189249, -0.0191748, -0.0189689, -0.0002204, 0.0002499
2: -0.0191893, -0.0188699, -0.0191666, -0.0188941, -0.0002952, 0.0002967
3: -0.0182902, -0.0179572, -0.0182717, -0.0179773, -0.0003129, 0.0003146
4: -0.0184495, -0.0179340, -0.0184131, -0.0179416, -0.0005079, 0.0004791

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202577, -0.0201233, -0.0001206, 0.0001396
1: -0.0191872, -0.0189515, -0.0191930, -0.0189217, -0.0002655, 0.0002415
2: -0.0191840, -0.0188793, -0.0191953, -0.0188504, -0.0003336, 0.0003160
3: -0.0182851, -0.0179569, -0.0182952, -0.0179459, -0.0003392, 0.0003383
4: -0.0184406, -0.0179232, -0.0184582, -0.0179009, -0.0005397, 0.0005350

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000717
time: 0.19 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000718
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202534, -0.0201245, -0.0001194, 0.0001352
1: -0.0191872, -0.0189515, -0.0191850, -0.0189358, -0.0002514, 0.0002335
2: -0.0191840, -0.0188793, -0.0191833, -0.0188694, -0.0003146, 0.0003040
3: -0.0182851, -0.0179569, -0.0182856, -0.0179656, -0.0003196, 0.0003287
4: -0.0184406, -0.0179232, -0.0184397, -0.0179284, -0.0005121, 0.0005165

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000748, upper bound: 0.0000723
time: 0.18 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000718
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202577, -0.0201233, -0.0001158, 0.0001370
1: -0.0191775, -0.0189624, -0.0191930, -0.0189217, -0.0002558, 0.0002306
2: -0.0191701, -0.0188988, -0.0191953, -0.0188504, -0.0003196, 0.0002965
3: -0.0182747, -0.0179704, -0.0182952, -0.0179459, -0.0003288, 0.0003248
4: -0.0184192, -0.0179553, -0.0184582, -0.0179009, -0.0005183, 0.0005029

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000716
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000719
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202534, -0.0201245, -0.0001146, 0.0001326
1: -0.0191775, -0.0189624, -0.0191850, -0.0189358, -0.0002417, 0.0002225
2: -0.0191701, -0.0188988, -0.0191833, -0.0188694, -0.0003006, 0.0002845
3: -0.0182747, -0.0179704, -0.0182856, -0.0179656, -0.0003092, 0.0003152
4: -0.0184192, -0.0179553, -0.0184397, -0.0179284, -0.0004907, 0.0004843

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000741, upper bound: 0.0000716
time: 0.19 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000755, upper bound: 0.0000724
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202414, -0.0201245, -0.0001194, 0.0001233
1: -0.0191872, -0.0189515, -0.0191837, -0.0189509, -0.0002363, 0.0002322
2: -0.0191840, -0.0188793, -0.0191792, -0.0188723, -0.0003117, 0.0002999
3: -0.0182851, -0.0179569, -0.0182814, -0.0179578, -0.0003274, 0.0003245
4: -0.0184406, -0.0179232, -0.0184330, -0.0179065, -0.0005340, 0.0005098

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202439, -0.0201182, -0.0202352, -0.0201252, -0.0001187, 0.0001171
1: -0.0191872, -0.0189515, -0.0191748, -0.0189689, -0.0002183, 0.0002233
2: -0.0191840, -0.0188793, -0.0191666, -0.0188941, -0.0002899, 0.0002873
3: -0.0182851, -0.0179569, -0.0182717, -0.0179773, -0.0003079, 0.0003148
4: -0.0184406, -0.0179232, -0.0184131, -0.0179416, -0.0004990, 0.0004899

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202414, -0.0201245, -0.0001146, 0.0001207
1: -0.0191775, -0.0189624, -0.0191837, -0.0189509, -0.0002266, 0.0002212
2: -0.0191701, -0.0188988, -0.0191792, -0.0188723, -0.0002978, 0.0002804
3: -0.0182747, -0.0179704, -0.0182814, -0.0179578, -0.0003169, 0.0003110
4: -0.0184192, -0.0179553, -0.0184330, -0.0179065, -0.0005126, 0.0004776

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202391, -0.0201208, -0.0202352, -0.0201252, -0.0001139, 0.0001144
1: -0.0191775, -0.0189624, -0.0191748, -0.0189689, -0.0002085, 0.0002123
2: -0.0191701, -0.0188988, -0.0191666, -0.0188941, -0.0002759, 0.0002678
3: -0.0182747, -0.0179704, -0.0182717, -0.0179773, -0.0002974, 0.0003013
4: -0.0184192, -0.0179553, -0.0184131, -0.0179416, -0.0004776, 0.0004578

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202577, -0.0201233, -0.0202439, -0.0201182, -0.0001396, 0.0001206
1: -0.0191930, -0.0189217, -0.0191872, -0.0189515, -0.0002415, 0.0002655
2: -0.0191953, -0.0188504, -0.0191840, -0.0188793, -0.0003160, 0.0003336
3: -0.0182952, -0.0179459, -0.0182851, -0.0179569, -0.0003383, 0.0003392
4: -0.0184582, -0.0179009, -0.0184406, -0.0179232, -0.0005350, 0.0005397

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202577, -0.0201233, -0.0202391, -0.0201208, -0.0001370, 0.0001158
1: -0.0191930, -0.0189217, -0.0191775, -0.0189624, -0.0002306, 0.0002558
2: -0.0191953, -0.0188504, -0.0191701, -0.0188988, -0.0002965, 0.0003196
3: -0.0182952, -0.0179459, -0.0182747, -0.0179704, -0.0003248, 0.0003288
4: -0.0184582, -0.0179009, -0.0184192, -0.0179553, -0.0005029, 0.0005183

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202534, -0.0201245, -0.0202439, -0.0201182, -0.0001352, 0.0001194
1: -0.0191850, -0.0189358, -0.0191872, -0.0189515, -0.0002335, 0.0002514
2: -0.0191833, -0.0188694, -0.0191840, -0.0188793, -0.0003040, 0.0003146
3: -0.0182856, -0.0179656, -0.0182851, -0.0179569, -0.0003287, 0.0003196
4: -0.0184397, -0.0179284, -0.0184406, -0.0179232, -0.0005165, 0.0005121

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202534, -0.0201245, -0.0202391, -0.0201208, -0.0001326, 0.0001146
1: -0.0191850, -0.0189358, -0.0191775, -0.0189624, -0.0002225, 0.0002417
2: -0.0191833, -0.0188694, -0.0191701, -0.0188988, -0.0002845, 0.0003006
3: -0.0182856, -0.0179656, -0.0182747, -0.0179704, -0.0003152, 0.0003092
4: -0.0184397, -0.0179284, -0.0184192, -0.0179553, -0.0004843, 0.0004907

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202624, -0.0201167, -0.0001247, 0.0001379
1: -0.0191837, -0.0189509, -0.0191982, -0.0189157, -0.0002680, 0.0002473
2: -0.0191792, -0.0188723, -0.0192029, -0.0188516, -0.0003276, 0.0003306
3: -0.0182814, -0.0179578, -0.0183009, -0.0179404, -0.0003410, 0.0003431
4: -0.0184330, -0.0179065, -0.0184696, -0.0179078, -0.0005252, 0.0005631

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202582, -0.0201194, -0.0001220, 0.0001337
1: -0.0191837, -0.0189509, -0.0191893, -0.0189249, -0.0002588, 0.0002384
2: -0.0191792, -0.0188723, -0.0191893, -0.0188699, -0.0003093, 0.0003170
3: -0.0182814, -0.0179578, -0.0182902, -0.0179572, -0.0003243, 0.0003324
4: -0.0184330, -0.0179065, -0.0184495, -0.0179340, -0.0004989, 0.0005429

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202624, -0.0201167, -0.0001185, 0.0001372
1: -0.0191748, -0.0189689, -0.0191982, -0.0189157, -0.0002591, 0.0002293
2: -0.0191666, -0.0188941, -0.0192029, -0.0188516, -0.0003150, 0.0003087
3: -0.0182717, -0.0179773, -0.0183009, -0.0179404, -0.0003313, 0.0003236
4: -0.0184131, -0.0179416, -0.0184696, -0.0179078, -0.0005054, 0.0005281

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202582, -0.0201194, -0.0001158, 0.0001330
1: -0.0191748, -0.0189689, -0.0191893, -0.0189249, -0.0002499, 0.0002204
2: -0.0191666, -0.0188941, -0.0191893, -0.0188699, -0.0002967, 0.0002952
3: -0.0182717, -0.0179773, -0.0182902, -0.0179572, -0.0003146, 0.0003129
4: -0.0184131, -0.0179416, -0.0184495, -0.0179340, -0.0004791, 0.0005079

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202439, -0.0201182, -0.0001233, 0.0001194
1: -0.0191837, -0.0189509, -0.0191872, -0.0189515, -0.0002322, 0.0002363
2: -0.0191792, -0.0188723, -0.0191840, -0.0188793, -0.0002999, 0.0003117
3: -0.0182814, -0.0179578, -0.0182851, -0.0179569, -0.0003245, 0.0003274
4: -0.0184330, -0.0179065, -0.0184406, -0.0179232, -0.0005098, 0.0005340

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202391, -0.0201208, -0.0001207, 0.0001146
1: -0.0191837, -0.0189509, -0.0191775, -0.0189624, -0.0002212, 0.0002266
2: -0.0191792, -0.0188723, -0.0191701, -0.0188988, -0.0002804, 0.0002978
3: -0.0182814, -0.0179578, -0.0182747, -0.0179704, -0.0003110, 0.0003169
4: -0.0184330, -0.0179065, -0.0184192, -0.0179553, -0.0004776, 0.0005126

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202439, -0.0201182, -0.0001171, 0.0001187
1: -0.0191748, -0.0189689, -0.0191872, -0.0189515, -0.0002233, 0.0002183
2: -0.0191666, -0.0188941, -0.0191840, -0.0188793, -0.0002873, 0.0002899
3: -0.0182717, -0.0179773, -0.0182851, -0.0179569, -0.0003148, 0.0003079
4: -0.0184131, -0.0179416, -0.0184406, -0.0179232, -0.0004899, 0.0004990

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202391, -0.0201208, -0.0001144, 0.0001139
1: -0.0191748, -0.0189689, -0.0191775, -0.0189624, -0.0002123, 0.0002085
2: -0.0191666, -0.0188941, -0.0191701, -0.0188988, -0.0002678, 0.0002759
3: -0.0182717, -0.0179773, -0.0182747, -0.0179704, -0.0003013, 0.0002974
4: -0.0184131, -0.0179416, -0.0184192, -0.0179553, -0.0004578, 0.0004776

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202577, -0.0201233, -0.0202414, -0.0201245, -0.0001332, 0.0001182
1: -0.0191930, -0.0189217, -0.0191837, -0.0189509, -0.0002421, 0.0002620
2: -0.0191953, -0.0188504, -0.0191792, -0.0188723, -0.0003230, 0.0003288
3: -0.0182952, -0.0179459, -0.0182814, -0.0179578, -0.0003374, 0.0003355
4: -0.0184582, -0.0179009, -0.0184330, -0.0179065, -0.0005517, 0.0005321

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202577, -0.0201233, -0.0202352, -0.0201252, -0.0001325, 0.0001119
1: -0.0191930, -0.0189217, -0.0191748, -0.0189689, -0.0002241, 0.0002531
2: -0.0191953, -0.0188504, -0.0191666, -0.0188941, -0.0003012, 0.0003162
3: -0.0182952, -0.0179459, -0.0182717, -0.0179773, -0.0003180, 0.0003258
4: -0.0184582, -0.0179009, -0.0184131, -0.0179416, -0.0005166, 0.0005123

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202534, -0.0201245, -0.0202414, -0.0201245, -0.0001289, 0.0001169
1: -0.0191850, -0.0189358, -0.0191837, -0.0189509, -0.0002341, 0.0002478
2: -0.0191833, -0.0188694, -0.0191792, -0.0188723, -0.0003110, 0.0003098
3: -0.0182856, -0.0179656, -0.0182814, -0.0179578, -0.0003279, 0.0003159
4: -0.0184397, -0.0179284, -0.0184330, -0.0179065, -0.0005331, 0.0005045

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202534, -0.0201245, -0.0202352, -0.0201252, -0.0001282, 0.0001107
1: -0.0191850, -0.0189358, -0.0191748, -0.0189689, -0.0002160, 0.0002390
2: -0.0191833, -0.0188694, -0.0191666, -0.0188941, -0.0002892, 0.0002972
3: -0.0182856, -0.0179656, -0.0182717, -0.0179773, -0.0003084, 0.0003062
4: -0.0184397, -0.0179284, -0.0184131, -0.0179416, -0.0004981, 0.0004847

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202577, -0.0201233, -0.0001182, 0.0001332
1: -0.0191837, -0.0189509, -0.0191930, -0.0189217, -0.0002620, 0.0002421
2: -0.0191792, -0.0188723, -0.0191953, -0.0188504, -0.0003288, 0.0003230
3: -0.0182814, -0.0179578, -0.0182952, -0.0179459, -0.0003355, 0.0003374
4: -0.0184330, -0.0179065, -0.0184582, -0.0179009, -0.0005321, 0.0005517

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202534, -0.0201245, -0.0001169, 0.0001289
1: -0.0191837, -0.0189509, -0.0191850, -0.0189358, -0.0002478, 0.0002341
2: -0.0191792, -0.0188723, -0.0191833, -0.0188694, -0.0003098, 0.0003110
3: -0.0182814, -0.0179578, -0.0182856, -0.0179656, -0.0003159, 0.0003279
4: -0.0184330, -0.0179065, -0.0184397, -0.0179284, -0.0005045, 0.0005331

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202577, -0.0201233, -0.0001119, 0.0001325
1: -0.0191748, -0.0189689, -0.0191930, -0.0189217, -0.0002531, 0.0002241
2: -0.0191666, -0.0188941, -0.0191953, -0.0188504, -0.0003162, 0.0003012
3: -0.0182717, -0.0179773, -0.0182952, -0.0179459, -0.0003258, 0.0003180
4: -0.0184131, -0.0179416, -0.0184582, -0.0179009, -0.0005123, 0.0005166

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202534, -0.0201245, -0.0001107, 0.0001282
1: -0.0191748, -0.0189689, -0.0191850, -0.0189358, -0.0002390, 0.0002160
2: -0.0191666, -0.0188941, -0.0191833, -0.0188694, -0.0002972, 0.0002892
3: -0.0182717, -0.0179773, -0.0182856, -0.0179656, -0.0003062, 0.0003084
4: -0.0184131, -0.0179416, -0.0184397, -0.0179284, -0.0004847, 0.0004981

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202414, -0.0201245, -0.0001169, 0.0001169
1: -0.0191837, -0.0189509, -0.0191837, -0.0189509, -0.0002328, 0.0002328
2: -0.0191792, -0.0188723, -0.0191792, -0.0188723, -0.0003069, 0.0003069
3: -0.0182814, -0.0179578, -0.0182814, -0.0179578, -0.0003236, 0.0003236
4: -0.0184330, -0.0179065, -0.0184330, -0.0179065, -0.0005264, 0.0005264

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202414, -0.0201245, -0.0202352, -0.0201252, -0.0001162, 0.0001107
1: -0.0191837, -0.0189509, -0.0191748, -0.0189689, -0.0002147, 0.0002239
2: -0.0191792, -0.0188723, -0.0191666, -0.0188941, -0.0002851, 0.0002943
3: -0.0182814, -0.0179578, -0.0182717, -0.0179773, -0.0003042, 0.0003139
4: -0.0184330, -0.0179065, -0.0184131, -0.0179416, -0.0004914, 0.0005066

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202414, -0.0201245, -0.0001107, 0.0001162
1: -0.0191748, -0.0189689, -0.0191837, -0.0189509, -0.0002239, 0.0002147
2: -0.0191666, -0.0188941, -0.0191792, -0.0188723, -0.0002943, 0.0002851
3: -0.0182717, -0.0179773, -0.0182814, -0.0179578, -0.0003139, 0.0003042
4: -0.0184131, -0.0179416, -0.0184330, -0.0179065, -0.0005066, 0.0004914

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202352, -0.0201252, -0.0202352, -0.0201252, -0.0001100, 0.0001100
1: -0.0191748, -0.0189689, -0.0191748, -0.0189689, -0.0002058, 0.0002058
2: -0.0191666, -0.0188941, -0.0191666, -0.0188941, -0.0002725, 0.0002725
3: -0.0182717, -0.0179773, -0.0182717, -0.0179773, -0.0002944, 0.0002944
4: -0.0184131, -0.0179416, -0.0184131, -0.0179416, -0.0004715, 0.0004715

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.73 + 167.75 = 169.48 seconds
