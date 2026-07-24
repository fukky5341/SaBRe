## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0055854


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005995, 0.0005995)
1: (0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033195, 0.0033195)
2: (0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0074161, 0.0074161)
3: (0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031252, 0.0031252)
4: (1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0121244, 0.0121244)
5: (0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023587, 0.0023587)
6: (-0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030695, 0.0030695)
7: (-0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003915, 0.0003915)
8: (-0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021208, 0.0021208)
9: (-0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0106170, 0.0106170)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.92 + 2.00 = 2.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0062060, upper bound: 0.0062060

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0061923, upper bound: 0.0058431
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058431, upper bound: 0.0061923
time: 1.05 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.17 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 4, lower bound: -0.0061923, upper bound: 0.0058431
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 4, lower bound: -0.0058431, upper bound: 0.0061923

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006044, 0.0006127
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033925, 0.0033466
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0074767, 0.0075792
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031939, 0.0031507
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0123912, 0.0122234
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0024106, 0.0023779
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030945, 0.0031370
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003947, 0.0004002
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021674, 0.0021381
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0107038, 0.0108506

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0060178, upper bound: 0.0056162
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059370, upper bound: 0.0056396
time: 1.09 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006127, 0.0006044
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033466, 0.0033925
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075792, 0.0074767
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031507, 0.0031939
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0122234, 0.0123912
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023779, 0.0024106
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031370, 0.0030945
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004002, 0.0003947
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021381, 0.0021674
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0108506, 0.0107038

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057631, upper bound: 0.0060950
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057529, upper bound: 0.0061117
time: 1.12 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.12 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 4, lower bound: -0.0060178, upper bound: 0.0056162
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 4, lower bound: -0.0059370, upper bound: 0.0056396
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 4, lower bound: -0.0057631, upper bound: 0.0060950
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 4, lower bound: -0.0057529, upper bound: 0.0061117

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005798, 0.0005909
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032717, 0.0032101
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0071718, 0.0073094
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030802, 0.0030222
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0119500, 0.0117250
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023247, 0.0022810
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029684, 0.0030253
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003786, 0.0003859
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020902, 0.0020509
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0102673, 0.0104643

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050032, upper bound: 0.0047713
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050032, upper bound: 0.0047713
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005821, 0.0005881
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032560, 0.0032233
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072012, 0.0072743
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030654, 0.0030346
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0118927, 0.0117731
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023136, 0.0022903
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029805, 0.0030108
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003802, 0.0003841
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020802, 0.0020593
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103094, 0.0104141

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059361, upper bound: 0.0056309
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0058831, upper bound: 0.0056387
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006102, 0.0006033
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033407, 0.0033785
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075480, 0.0074635
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031451, 0.0031808
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0122019, 0.0123401
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023738, 0.0024006
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031241, 0.0030891
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003985, 0.0003940
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021343, 0.0021585
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0108059, 0.0106849

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 238

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056757, upper bound: 0.0059941
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056743, upper bound: 0.0060000
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006127, 0.0006019
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033326, 0.0033925
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075792, 0.0074454
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031375, 0.0031939
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0121724, 0.0123912
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023680, 0.0024106
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031370, 0.0030816
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004002, 0.0003931
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021292, 0.0021674
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0108506, 0.0106591

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057204, upper bound: 0.0060793
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057187, upper bound: 0.0060793
time: 1.06 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.39 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.39
Output dim: 4, lower bound: -0.0050032, upper bound: 0.0047713
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.39
Output dim: 4, lower bound: -0.0050032, upper bound: 0.0047713
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 4, lower bound: -0.0059361, upper bound: 0.0056309
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 4, lower bound: -0.0058831, upper bound: 0.0056387
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 4, lower bound: -0.0056757, upper bound: 0.0059941
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 4, lower bound: -0.0056743, upper bound: 0.0060000
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 4, lower bound: -0.0057204, upper bound: 0.0060793
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 4, lower bound: -0.0057187, upper bound: 0.0060793

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005852, 0.0005936
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032868, 0.0032403
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072392, 0.0073432
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030944, 0.0030506
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0120052, 0.0118353
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023355, 0.0023024
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029963, 0.0030393
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003822, 0.0003877
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020999, 0.0020702
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103638, 0.0105127

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057412, upper bound: 0.0052966
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055888, upper bound: 0.0054295
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005878, 0.0005911
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032731, 0.0032547
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072713, 0.0073124
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030815, 0.0030642
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0119549, 0.0118878
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023257, 0.0023126
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030096, 0.0030266
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003839, 0.0003861
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020911, 0.0020794
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0104098, 0.0104686

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056743, upper bound: 0.0053002
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055452, upper bound: 0.0054421
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006069, 0.0006007
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033261, 0.0033604
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075075, 0.0074308
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031313, 0.0031637
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0121484, 0.0122738
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023634, 0.0023877
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031073, 0.0030756
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003964, 0.0003923
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021250, 0.0021469
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0107479, 0.0106381

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 136

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051907, upper bound: 0.0054559
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051907, upper bound: 0.0054559
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006075, 0.0006033
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033407, 0.0033639
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075153, 0.0074635
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031451, 0.0031670
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0122019, 0.0122867
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023738, 0.0023902
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031106, 0.0030891
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003968, 0.0003940
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021343, 0.0021491
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0107591, 0.0106849

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056735, upper bound: 0.0059268
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056650, upper bound: 0.0059992
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006222, 0.0006099
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033768, 0.0034449
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076964, 0.0075441
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031791, 0.0032433
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0123336, 0.0125827
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023994, 0.0024478
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031855, 0.0031224
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004063, 0.0003983
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021573, 0.0022009
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0110183, 0.0108002

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053659, upper bound: 0.0056732
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053659, upper bound: 0.0056732
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006207, 0.0006113
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033848, 0.0034368
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076781, 0.0075619
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031866, 0.0032356
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0123628, 0.0125528
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0024051, 0.0024420
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031779, 0.0031298
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004054, 0.0003992
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021625, 0.0021957
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0109922, 0.0108258

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057187, upper bound: 0.0060144
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056873, upper bound: 0.0060793
time: 1.64 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.49 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0057412, upper bound: 0.0052966
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0055888, upper bound: 0.0054295
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0056743, upper bound: 0.0053002
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0055452, upper bound: 0.0054421
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0051907, upper bound: 0.0054559
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0051907, upper bound: 0.0054559
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0056735, upper bound: 0.0059268
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0056650, upper bound: 0.0059992
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0053659, upper bound: 0.0056732
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0053659, upper bound: 0.0056732
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0057187, upper bound: 0.0060144
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.49
Output dim: 4, lower bound: -0.0056873, upper bound: 0.0060793

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005708, 0.0005818
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032215, 0.0031605
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0070609, 0.0071972
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030329, 0.0029755
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0117665, 0.0115437
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022891, 0.0022457
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029225, 0.0029789
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003728, 0.0003800
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020582, 0.0020192
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0101085, 0.0103037

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047518, upper bound: 0.0044370
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047518, upper bound: 0.0044370
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005734, 0.0005792
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032070, 0.0031748
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0070929, 0.0071648
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030193, 0.0029890
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0117136, 0.0115960
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022788, 0.0022559
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029357, 0.0029655
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003745, 0.0003783
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020489, 0.0020283
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0101544, 0.0102573

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051557, upper bound: 0.0051886
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0053277, upper bound: 0.0049769
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005734, 0.0005791
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032065, 0.0031749
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0070930, 0.0071638
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030188, 0.0029890
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0117119, 0.0115962
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022784, 0.0022559
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029357, 0.0029650
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003745, 0.0003782
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020486, 0.0020284
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0101545, 0.0102558

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055918, upper bound: 0.0052109
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055601, upper bound: 0.0052165
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006094, 0.0006074
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033633, 0.0033743
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075385, 0.0075141
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031665, 0.0031767
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0122846, 0.0123245
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023899, 0.0023976
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031201, 0.0031100
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003980, 0.0003967
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021488, 0.0021558
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0107922, 0.0107574

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054747, upper bound: 0.0056825
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054522, upper bound: 0.0057480
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006119, 0.0006052
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033508, 0.0033880
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075692, 0.0074861
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031546, 0.0031897
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0122388, 0.0123748
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023809, 0.0024074
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031329, 0.0030984
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003996, 0.0003952
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021408, 0.0021646
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0108363, 0.0107172

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056310, upper bound: 0.0059640
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0056303, upper bound: 0.0059640
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006196, 0.0006058
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033542, 0.0034304
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076640, 0.0074937
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031579, 0.0032296
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0122513, 0.0125297
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023834, 0.0024375
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031721, 0.0031016
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004046, 0.0003956
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021429, 0.0021916
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0109719, 0.0107281

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051507, upper bound: 0.0052913
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049932, upper bound: 0.0054712
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006181, 0.0006099
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033768, 0.0034224
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076460, 0.0075441
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031791, 0.0032221
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0123336, 0.0125003
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023994, 0.0024318
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031646, 0.0031224
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004037, 0.0003983
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021573, 0.0021865
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0109462, 0.0108002

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053659, upper bound: 0.0056021
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053324, upper bound: 0.0056732
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006207, 0.0006118
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033873, 0.0034370
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076787, 0.0075676
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031890, 0.0032358
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0123721, 0.0125537
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0024069, 0.0024422
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031782, 0.0031322
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004054, 0.0003995
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021641, 0.0021958
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0109930, 0.0108339

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052598, upper bound: 0.0057308
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054766, upper bound: 0.0056137
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006211, 0.0006113
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033850, 0.0034392
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076836, 0.0075625
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031869, 0.0032379
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0123638, 0.0125618
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0024053, 0.0024438
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031802, 0.0031301
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004057, 0.0003993
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021626, 0.0021973
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0110000, 0.0108267

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054847, upper bound: 0.0057227
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053589, upper bound: 0.0058824
time: 1.70 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.21 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0047518, upper bound: 0.0044370
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0047518, upper bound: 0.0044370
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0051557, upper bound: 0.0051886
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0053277, upper bound: 0.0049769
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0055918, upper bound: 0.0052109
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0055601, upper bound: 0.0052165
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0054747, upper bound: 0.0056825
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0054522, upper bound: 0.0057480
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0056310, upper bound: 0.0059640
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0056303, upper bound: 0.0059640
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0051507, upper bound: 0.0052913
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0049932, upper bound: 0.0054712
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0053659, upper bound: 0.0056021
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0053324, upper bound: 0.0056732
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0052598, upper bound: 0.0057308
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0054766, upper bound: 0.0056137
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0054847, upper bound: 0.0057227
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 4, lower bound: -0.0053589, upper bound: 0.0058824

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005708, 0.0005779
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032000, 0.0031604
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0070607, 0.0071492
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030127, 0.0029754
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0116880, 0.0115434
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022738, 0.0022457
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029224, 0.0029590
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003728, 0.0003774
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020444, 0.0020191
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0101083, 0.0102349

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055524, upper bound: 0.0051713
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055592, upper bound: 0.0051800
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005858, 0.0005863
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032464, 0.0032436
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072466, 0.0072527
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030563, 0.0030537
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0118573, 0.0118473
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023067, 0.0023048
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029993, 0.0030019
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003826, 0.0003829
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020740, 0.0020723
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103744, 0.0103832

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054747, upper bound: 0.0056414
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054379, upper bound: 0.0056825
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005886, 0.0005840
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032335, 0.0032593
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072816, 0.0072239
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030442, 0.0030685
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0118103, 0.0119046
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022976, 0.0023159
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030138, 0.0029899
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003844, 0.0003814
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020658, 0.0020823
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0104246, 0.0103420

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 216

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053277, upper bound: 0.0056294
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053333, upper bound: 0.0056228
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006207, 0.0006126
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033917, 0.0034371
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076788, 0.0075774
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031931, 0.0032359
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0123881, 0.0125539
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0024100, 0.0024422
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031782, 0.0031362
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004054, 0.0004001
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021669, 0.0021959
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0109931, 0.0108480

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047433, upper bound: 0.0049827
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047433, upper bound: 0.0049827
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006193, 0.0006140
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033997, 0.0034289
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076604, 0.0075953
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0032007, 0.0032281
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0124173, 0.0125239
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0024157, 0.0024364
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031706, 0.0031436
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004044, 0.0004010
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021720, 0.0021906
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0109669, 0.0108735

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047435, upper bound: 0.0049825
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047435, upper bound: 0.0049825
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006181, 0.0006103
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033793, 0.0034225
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076464, 0.0075498
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031815, 0.0032222
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0123429, 0.0125009
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0024012, 0.0024319
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031648, 0.0031248
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004037, 0.0003986
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021590, 0.0021866
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0109467, 0.0108084

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0052807, upper bound: 0.0055545
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0053207, upper bound: 0.0055241
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006185, 0.0006099
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033770, 0.0034249
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076515, 0.0075446
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031793, 0.0032244
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0123346, 0.0125093
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023996, 0.0024336
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031669, 0.0031227
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004040, 0.0003983
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021575, 0.0021881
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0109541, 0.0108011

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053314, upper bound: 0.0055926
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053220, upper bound: 0.0056724
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006042, 0.0005893
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032627, 0.0033457
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0074746, 0.0072892
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030717, 0.0031498
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0119169, 0.0122201
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023183, 0.0023773
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030937, 0.0030169
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003946, 0.0003848
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020845, 0.0021375
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0107008, 0.0104353

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052590, upper bound: 0.0057034
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052372, upper bound: 0.0057299
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005982, 0.0005966
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033034, 0.0033124
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0074002, 0.0073802
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031100, 0.0031185
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0120658, 0.0120985
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023473, 0.0023536
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030629, 0.0030546
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003907, 0.0003896
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021105, 0.0021162
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0105944, 0.0105657

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0054758, upper bound: 0.0055566
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0054727, upper bound: 0.0056128
time: 1.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006078, 0.0006007
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033258, 0.0033653
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075184, 0.0074303
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031311, 0.0031683
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0121476, 0.0122918
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023632, 0.0023912
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031118, 0.0030753
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003969, 0.0003923
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021248, 0.0021500
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0107636, 0.0106373

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0052807, upper bound: 0.0054761
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0052571, upper bound: 0.0055372
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006102, 0.0005980
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033110, 0.0033787
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075483, 0.0073971
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031171, 0.0031809
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0120933, 0.0123406
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023526, 0.0024007
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031242, 0.0030616
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003985, 0.0003905
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021153, 0.0021586
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0108064, 0.0105898

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 238

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052761, upper bound: 0.0057709
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052755, upper bound: 0.0057770
time: 1.55 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.70 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0055524, upper bound: 0.0051713
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0055592, upper bound: 0.0051800
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0054747, upper bound: 0.0056414
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0054379, upper bound: 0.0056825
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0053277, upper bound: 0.0056294
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0053333, upper bound: 0.0056228
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0047433, upper bound: 0.0049827
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0047433, upper bound: 0.0049827
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0047435, upper bound: 0.0049825
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0047435, upper bound: 0.0049825
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0052807, upper bound: 0.0055545
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0053207, upper bound: 0.0055241
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0053314, upper bound: 0.0055926
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0053220, upper bound: 0.0056724
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0052590, upper bound: 0.0057034
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0052372, upper bound: 0.0057299
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0054758, upper bound: 0.0055566
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0054727, upper bound: 0.0056128
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0052807, upper bound: 0.0054761
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0052571, upper bound: 0.0055372
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0052761, upper bound: 0.0057709
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 4, lower bound: -0.0052755, upper bound: 0.0057770

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005863, 0.0005873
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032518, 0.0032466
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072532, 0.0072649
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030615, 0.0030565
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0118773, 0.0118580
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023106, 0.0023069
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030020, 0.0030069
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003829, 0.0003836
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020775, 0.0020742
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103838, 0.0104007

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050412, upper bound: 0.0053902
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0052221, upper bound: 0.0051879
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005867, 0.0005868
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032492, 0.0032488
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072581, 0.0072592
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030590, 0.0030586
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0118679, 0.0118661
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023088, 0.0023084
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030041, 0.0030045
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003832, 0.0003833
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020759, 0.0020756
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103909, 0.0103924

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045657, upper bound: 0.0047424
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0045657, upper bound: 0.0047424
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005814, 0.0005772
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031961, 0.0032190
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0071916, 0.0071404
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030090, 0.0030306
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0116737, 0.0117574
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022710, 0.0022873
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029766, 0.0029554
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003797, 0.0003770
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020419, 0.0020566
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0102957, 0.0102224

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053277, upper bound: 0.0055872
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052923, upper bound: 0.0056293
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005821, 0.0005763
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031910, 0.0032230
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072005, 0.0071291
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030042, 0.0030343
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0116553, 0.0117720
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022674, 0.0022901
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029803, 0.0029507
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003802, 0.0003764
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020387, 0.0020591
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103084, 0.0102062

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0053333, upper bound: 0.0055843
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0052997, upper bound: 0.0056228
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006200, 0.0006138
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033986, 0.0034331
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0076700, 0.0075928
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031996, 0.0032322
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0124133, 0.0125395
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0024149, 0.0024394
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031746, 0.0031426
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004049, 0.0004009
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021713, 0.0021934
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0109805, 0.0108700

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 238

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047398, upper bound: 0.0049300
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0047398, upper bound: 0.0049300
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006225, 0.0006114
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033855, 0.0034470
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0077011, 0.0075635
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031873, 0.0032452
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0123654, 0.0125903
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0024056, 0.0024493
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031874, 0.0031305
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0004066, 0.0003993
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021629, 0.0022023
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0110250, 0.0108281

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050959, upper bound: 0.0052904
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049745, upper bound: 0.0054704
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006054, 0.0005928
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032822, 0.0033522
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0074891, 0.0073327
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030900, 0.0031559
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0119882, 0.0122438
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023322, 0.0023819
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030997, 0.0030350
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003954, 0.0003871
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020969, 0.0021416
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0107216, 0.0104977

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050605, upper bound: 0.0053861
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049164, upper bound: 0.0054744
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006077, 0.0005904
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032693, 0.0033649
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075175, 0.0073039
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030779, 0.0031679
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0119410, 0.0122902
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023230, 0.0023909
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031115, 0.0030230
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003969, 0.0003856
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020887, 0.0021498
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0107622, 0.0104564

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050455, upper bound: 0.0054929
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0049995, upper bound: 0.0055389
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006019, 0.0005978
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033100, 0.0033327
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0074455, 0.0073949
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031163, 0.0031376
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0120899, 0.0121725
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023520, 0.0023680
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030817, 0.0030607
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003931, 0.0003904
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021147, 0.0021292
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0106592, 0.0105868

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050931, upper bound: 0.0051932
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050931, upper bound: 0.0051932
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006069, 0.0005954
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032967, 0.0033603
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075072, 0.0073653
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031038, 0.0031636
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0120414, 0.0122734
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023425, 0.0023877
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031072, 0.0030485
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003964, 0.0003889
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021062, 0.0021468
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0107475, 0.0105443

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0043960, upper bound: 0.0047618
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0043960, upper bound: 0.0047618
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0006076, 0.0005980
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033110, 0.0033645
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0075166, 0.0073971
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031171, 0.0031675
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0120933, 0.0122887
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023526, 0.0023906
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0031111, 0.0030616
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003968, 0.0003905
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021153, 0.0021495
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0107609, 0.0105898

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048204, upper bound: 0.0055125
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050200, upper bound: 0.0053529
time: 1.54 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.43 seconds
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0050412, upper bound: 0.0053902
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0052221, upper bound: 0.0051879
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0045657, upper bound: 0.0047424
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0045657, upper bound: 0.0047424
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0053277, upper bound: 0.0055872
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0052923, upper bound: 0.0056293
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0053333, upper bound: 0.0055843
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0052997, upper bound: 0.0056228
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0047398, upper bound: 0.0049300
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0047398, upper bound: 0.0049300
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0050959, upper bound: 0.0052904
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0049745, upper bound: 0.0054704
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0050605, upper bound: 0.0053861
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0049164, upper bound: 0.0054744
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0050455, upper bound: 0.0054929
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0049995, upper bound: 0.0055389
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0050931, upper bound: 0.0051932
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0050931, upper bound: 0.0051932
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0043960, upper bound: 0.0047618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0043960, upper bound: 0.0047618
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0048204, upper bound: 0.0055125
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 4, lower bound: -0.0050200, upper bound: 0.0053529

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005819, 0.0005782
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0032017, 0.0032222
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0071987, 0.0071529
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030142, 0.0030336
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0116941, 0.0117690
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022750, 0.0022895
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029795, 0.0029605
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003801, 0.0003776
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020455, 0.0020586
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103058, 0.0102402

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0052556, upper bound: 0.0055392
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0052814, upper bound: 0.0055098
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005824, 0.0005778
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031992, 0.0032246
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072040, 0.0071474
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030119, 0.0030358
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0116851, 0.0117777
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022732, 0.0022912
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029817, 0.0029583
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003803, 0.0003774
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020439, 0.0020601
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103135, 0.0102324

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044727, upper bound: 0.0046764
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0044727, upper bound: 0.0046764
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005831, 0.0005769
1: 0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0031942, 0.0032285
2: 0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0072129, 0.0071361
3: 0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0030072, 0.0030395
4: 1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0116667, 0.0117923
5: 0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0022696, 0.0022941
6: -0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0029854, 0.0029536
7: -0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003808, 0.0003768
8: -0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0020407, 0.0020627
9: -0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0103262, 0.0102162

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0048552, upper bound: 0.0053565
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0050529, upper bound: 0.0052033
time: 1.49 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.05 seconds
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.05
Output dim: 4, lower bound: -0.0052556, upper bound: 0.0055392
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.05
Output dim: 4, lower bound: -0.0052814, upper bound: 0.0055098
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.05
Output dim: 4, lower bound: -0.0044727, upper bound: 0.0046764
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.05
Output dim: 4, lower bound: -0.0044727, upper bound: 0.0046764
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.05
Output dim: 4, lower bound: -0.0048552, upper bound: 0.0053565
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.05
Output dim: 4, lower bound: -0.0050529, upper bound: 0.0052033

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.92 + 160.77 = 163.69 seconds
