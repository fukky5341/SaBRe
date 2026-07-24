## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0018531149999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791)
1: (-0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634)
2: (0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563)
3: (-0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303)
4: (-0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950)
5: (0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163)
6: (0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807)
7: (-0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978)
8: (0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931)
9: (0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.69 + 2.23 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0022462, upper bound: 0.0022461

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022193, upper bound: 0.0022316
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022316, upper bound: 0.0022193
time: 0.98 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.98 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.98
Output dim: 2, lower bound: -0.0022193, upper bound: 0.0022316
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.98
Output dim: 2, lower bound: -0.0022316, upper bound: 0.0022193

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020692, upper bound: 0.0022231
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022107, upper bound: 0.0020849
time: 0.98 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021704, upper bound: 0.0021513
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021645, upper bound: 0.0021587
time: 1.00 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.69 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 2, lower bound: -0.0020692, upper bound: 0.0022231
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 2, lower bound: -0.0022107, upper bound: 0.0020849
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 2, lower bound: -0.0021704, upper bound: 0.0021513
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 2, lower bound: -0.0021645, upper bound: 0.0021587

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000773, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028954, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034746, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0256278, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019491
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0019700
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009582, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0066417
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0052692
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0094771

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020683, upper bound: 0.0022222
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020683, upper bound: 0.0022222
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000773
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0028962
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0034756
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0256353
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019497, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019705, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009585
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0066436, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052707, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094799, 0.0096999

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021935, upper bound: 0.0020824
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022082, upper bound: 0.0020704
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021539, upper bound: 0.0021483
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0021679, upper bound: 0.0021319
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020131, upper bound: 0.0020057
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020131, upper bound: 0.0020057
time: 1.31 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 2, lower bound: -0.0020683, upper bound: 0.0022222
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 2, lower bound: -0.0020683, upper bound: 0.0022222
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 2, lower bound: -0.0021935, upper bound: 0.0020824
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 2, lower bound: -0.0022082, upper bound: 0.0020704
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 2, lower bound: -0.0021539, upper bound: 0.0021483
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 2, lower bound: -0.0021679, upper bound: 0.0021319
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 2, lower bound: -0.0020131, upper bound: 0.0020057
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 2, lower bound: -0.0020131, upper bound: 0.0020057

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000779, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029164, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034998, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0258141, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019633
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0019843
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009651, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0066899
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053075
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0095460

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020044, upper bound: 0.0021611
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020086, upper bound: 0.0021606
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000778, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029138, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034966, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0257906, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019615
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0019825
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009643, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0066839
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053027
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0095374

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020046, upper bound: 0.0021611
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020086, upper bound: 0.0021606
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000785, 0.0000768
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029395, 0.0028745
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035276, 0.0034495
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0260187, 0.0254432
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019351, 0.0019789
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019558, 0.0020000
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009728, 0.0009513
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065938, 0.0067430
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052312, 0.0053496
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094089, 0.0096217

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020704, upper bound: 0.0019413
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020704, upper bound: 0.0019410
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000787, 0.0000766
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029453, 0.0028693
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035345, 0.0034433
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0260700, 0.0253974
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019316, 0.0019828
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019522, 0.0020039
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009747, 0.0009496
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065820, 0.0067563
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052218, 0.0053601
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093919, 0.0096406

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017071, upper bound: 0.0015983
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017071, upper bound: 0.0015983
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015215, upper bound: 0.0015254
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015215, upper bound: 0.0015254
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020460, upper bound: 0.0020222
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020460, upper bound: 0.0020222
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015481, upper bound: 0.0015480
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015481, upper bound: 0.0015480
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018679, upper bound: 0.0019985
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020060, upper bound: 0.0018606
time: 1.35 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.23 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0020044, upper bound: 0.0021611
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0020086, upper bound: 0.0021606
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0020046, upper bound: 0.0021611
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0020086, upper bound: 0.0021606
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0020704, upper bound: 0.0019413
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0020704, upper bound: 0.0019410
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0017071, upper bound: 0.0015983
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0017071, upper bound: 0.0015983
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0015215, upper bound: 0.0015254
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0015215, upper bound: 0.0015254
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0020460, upper bound: 0.0020222
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0020460, upper bound: 0.0020222
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0015481, upper bound: 0.0015480
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0015481, upper bound: 0.0015480
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0018679, upper bound: 0.0019985
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.23
Output dim: 2, lower bound: -0.0020060, upper bound: 0.0018606

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000772, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028901, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034683, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0255815, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019456
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0019664
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009565, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0066297
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0052597
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0094600

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019939, upper bound: 0.0021583
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020013, upper bound: 0.0021427
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000779, 0.0000790
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029164, 0.0029600
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034998, 0.0035521
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0258141, 0.0261996
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019926, 0.0019633
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020139, 0.0019843
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009651, 0.0009796
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067899, 0.0066899
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053868, 0.0053075
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096886, 0.0095460

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018952, upper bound: 0.0020489
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018953, upper bound: 0.0020493
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000771, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028875, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034651, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0255581, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019438
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0019646
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009556, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0066236
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0052548
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0094513

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018899, upper bound: 0.0020500
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018899, upper bound: 0.0020506
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000778, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029138, 0.0029627
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034966, 0.0035553
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0257906, 0.0262235
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019944, 0.0019615
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020157, 0.0019825
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009643, 0.0009805
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067960, 0.0066839
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053917, 0.0053027
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096974, 0.0095374

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018136, upper bound: 0.0019498
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018136, upper bound: 0.0019498
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000785, 0.0000768
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029392, 0.0028748
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035272, 0.0034499
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0260161, 0.0254461
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019353, 0.0019787
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019560, 0.0019998
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009727, 0.0009514
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065946, 0.0067423
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052318, 0.0053490
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094100, 0.0096207

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019363, upper bound: 0.0018175
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019363, upper bound: 0.0018175
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000785, 0.0000768
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029395, 0.0028742
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035276, 0.0034492
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0260187, 0.0254406
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019349, 0.0019789
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019556, 0.0020000
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009728, 0.0009512
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065932, 0.0067430
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052307, 0.0053496
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094079, 0.0096217

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019710, upper bound: 0.0018474
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019699, upper bound: 0.0018474
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016551, upper bound: 0.0016416
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016551, upper bound: 0.0016416
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029634, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035563, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262303, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019950
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0020163
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009807, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0067978
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0053931
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0096999

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013836, upper bound: 0.0013813
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013836, upper bound: 0.0013813
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000765, 0.0000788
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028643, 0.0029514
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034373, 0.0035419
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253526, 0.0261241
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019869, 0.0019282
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020081, 0.0019488
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009479, 0.0009767
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067703, 0.0065704
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053712, 0.0052126
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096607, 0.0093754

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017261, upper bound: 0.0018477
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017261, upper bound: 0.0018477
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000784, 0.0000769
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029359, 0.0028798
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035232, 0.0034559
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0259866, 0.0254897
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019386, 0.0019764
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019593, 0.0019975
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009716, 0.0009530
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0066059, 0.0067347
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052408, 0.0053430
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094261, 0.0096098

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018907, upper bound: 0.0017445
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018907, upper bound: 0.0017445
time: 0.97 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0019939, upper bound: 0.0021583
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0020013, upper bound: 0.0021427
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0018952, upper bound: 0.0020489
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0018953, upper bound: 0.0020493
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0018899, upper bound: 0.0020500
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0018899, upper bound: 0.0020506
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0018136, upper bound: 0.0019498
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0018136, upper bound: 0.0019498
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0019363, upper bound: 0.0018175
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0019363, upper bound: 0.0018175
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0019710, upper bound: 0.0018474
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0019699, upper bound: 0.0018474
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0016551, upper bound: 0.0016416
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0016551, upper bound: 0.0016416
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0013836, upper bound: 0.0013813
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0013836, upper bound: 0.0013813
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0017261, upper bound: 0.0018477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0017261, upper bound: 0.0018477
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0018907, upper bound: 0.0017445
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 2, lower bound: -0.0018907, upper bound: 0.0017445

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000765, 0.0000786
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028634, 0.0029447
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034362, 0.0035338
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253450, 0.0260648
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019824, 0.0019276
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020035, 0.0019482
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009476, 0.0009745
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067549, 0.0065684
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053590, 0.0052110
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096387, 0.0093725

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018371, upper bound: 0.0019928
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018371, upper bound: 0.0019928
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000766, 0.0000785
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028687, 0.0029389
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034426, 0.0035268
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253920, 0.0260129
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019784, 0.0019312
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019996, 0.0019518
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009494, 0.0009726
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067415, 0.0065806
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053484, 0.0052207
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096195, 0.0093899

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017649, upper bound: 0.0018949
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017649, upper bound: 0.0018949
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000729, 0.0000747
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027307, 0.0027963
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032770, 0.0033557
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0241705, 0.0247508
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018824, 0.0018383
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019025, 0.0018579
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009037, 0.0009254
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0064144, 0.0062640
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050889, 0.0049696
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0091528, 0.0089382

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018137, upper bound: 0.0019700
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018129, upper bound: 0.0019724
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000735, 0.0000741
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027516, 0.0027766
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0033020, 0.0033321
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0243552, 0.0245769
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018692, 0.0018524
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018892, 0.0018721
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009106, 0.0009189
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0063693, 0.0063119
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050531, 0.0050075
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0090885, 0.0090065

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018858, upper bound: 0.0020397
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018856, upper bound: 0.0020384
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000722, 0.0000749
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027020, 0.0028053
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032425, 0.0033665
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0239160, 0.0248308
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018885, 0.0018190
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019087, 0.0018384
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008942, 0.0009284
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0064351, 0.0061980
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0051053, 0.0049172
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0091824, 0.0088441

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016994, upper bound: 0.0018378
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016994, upper bound: 0.0018378
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000727, 0.0000744
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027238, 0.0027869
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032687, 0.0033444
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0241093, 0.0246675
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018761, 0.0018337
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018961, 0.0018532
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009014, 0.0009223
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0063928, 0.0062481
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050717, 0.0049570
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0091220, 0.0089156

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018804, upper bound: 0.0020411
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018804, upper bound: 0.0020402
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000778, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029135, 0.0029629
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034963, 0.0035556
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0257880, 0.0262258
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019946, 0.0019613
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020159, 0.0019823
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009642, 0.0009805
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067966, 0.0066832
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053921, 0.0053021
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096983, 0.0095364

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014811, upper bound: 0.0015650
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014811, upper bound: 0.0015650
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000778, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029138, 0.0029624
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034966, 0.0035550
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0257906, 0.0262208
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019942, 0.0019615
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020155, 0.0019825
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009643, 0.0009804
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067954, 0.0066839
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053911, 0.0053027
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096964, 0.0095374

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016994, upper bound: 0.0018377
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016994, upper bound: 0.0018378
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000778, 0.0000763
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029139, 0.0028576
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034968, 0.0034293
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0257919, 0.0252937
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019237, 0.0019616
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019443, 0.0019826
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009643, 0.0009457
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065551, 0.0066842
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052005, 0.0053029
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093536, 0.0095378

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019355, upper bound: 0.0018164
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019354, upper bound: 0.0018167
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000785, 0.0000761
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029392, 0.0028495
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035272, 0.0034195
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0260161, 0.0252219
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019183, 0.0019787
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019388, 0.0019998
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009727, 0.0009430
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065365, 0.0067423
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0051857, 0.0053490
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093271, 0.0096207

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019355, upper bound: 0.0018164
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019354, upper bound: 0.0018167
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000775, 0.0000759
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029025, 0.0028424
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034831, 0.0034110
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0256907, 0.0251591
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019135, 0.0019539
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019339, 0.0019748
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009605, 0.0009407
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065202, 0.0066580
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0051728, 0.0052821
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093038, 0.0095004

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018646, upper bound: 0.0017364
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018623, upper bound: 0.0017368
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000776, 0.0000768
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029077, 0.0028742
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034894, 0.0034492
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0257373, 0.0254406
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019349, 0.0019575
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019556, 0.0019784
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009623, 0.0009512
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065932, 0.0066700
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052307, 0.0052917
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094079, 0.0095176

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018669, upper bound: 0.0016752
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018002, upper bound: 0.0017417
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000735, 0.0000725
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027508, 0.0027151
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0033010, 0.0032582
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0243477, 0.0240321
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018278, 0.0018518
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018473, 0.0018716
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009103, 0.0008985
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062281, 0.0063099
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049411, 0.0050060
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0088871, 0.0090038

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018519, upper bound: 0.0016897
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018417, upper bound: 0.0017054
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000740, 0.0000719
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027704, 0.0026933
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0033246, 0.0032320
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0245217, 0.0238389
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018131, 0.0018650
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018324, 0.0018849
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009168, 0.0008913
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0061781, 0.0063550
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049014, 0.0050418
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0088156, 0.0090681

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017363, upper bound: 0.0015971
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017363, upper bound: 0.0015971
time: 1.38 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018371, upper bound: 0.0019928
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018371, upper bound: 0.0019928
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0017649, upper bound: 0.0018949
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0017649, upper bound: 0.0018949
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018137, upper bound: 0.0019700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018129, upper bound: 0.0019724
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018858, upper bound: 0.0020397
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018856, upper bound: 0.0020384
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0016994, upper bound: 0.0018378
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0016994, upper bound: 0.0018378
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018804, upper bound: 0.0020411
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018804, upper bound: 0.0020402
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0014811, upper bound: 0.0015650
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0014811, upper bound: 0.0015650
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0016994, upper bound: 0.0018377
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0016994, upper bound: 0.0018378
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0019355, upper bound: 0.0018164
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0019354, upper bound: 0.0018167
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0019355, upper bound: 0.0018164
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0019354, upper bound: 0.0018167
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018646, upper bound: 0.0017364
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018623, upper bound: 0.0017368
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018669, upper bound: 0.0016752
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018002, upper bound: 0.0017417
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018519, upper bound: 0.0016897
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0018417, upper bound: 0.0017054
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0017363, upper bound: 0.0015971
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.37
Output dim: 2, lower bound: -0.0017363, upper bound: 0.0015971

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000760, 0.0000783
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028478, 0.0029317
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034175, 0.0035182
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0252072, 0.0259496
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019736, 0.0019172
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019947, 0.0019376
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009425, 0.0009702
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067251, 0.0065327
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053354, 0.0051827
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0095962, 0.0093216

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017919, upper bound: 0.0019373
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017758, upper bound: 0.0019475
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000765, 0.0000782
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028634, 0.0029292
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034362, 0.0035151
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253450, 0.0259271
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019719, 0.0019276
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019930, 0.0019482
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009476, 0.0009694
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067192, 0.0065684
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053307, 0.0052110
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0095878, 0.0093725

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013751, upper bound: 0.0014333
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013751, upper bound: 0.0014333
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000756, 0.0000791
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028323, 0.0029634
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0033989, 0.0035563
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0250696, 0.0262303
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019950, 0.0019067
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0020163, 0.0019270
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009373, 0.0009807
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0067978, 0.0064970
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0053931, 0.0051544
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0096999, 0.0092707

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016576, upper bound: 0.0017363
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015985, upper bound: 0.0017903
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000766, 0.0000775
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028687, 0.0029024
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0034426, 0.0034831
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0253920, 0.0256905
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019539, 0.0019312
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019748, 0.0019518
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009494, 0.0009605
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0066579, 0.0065806
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052821, 0.0052207
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0095003, 0.0093899

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013838, upper bound: 0.0014698
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013838, upper bound: 0.0014698
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000720, 0.0000738
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026967, 0.0027650
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032362, 0.0033181
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0238694, 0.0244740
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018614, 0.0018154
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018813, 0.0018348
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008924, 0.0009150
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0063427, 0.0061860
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050320, 0.0049077
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0090505, 0.0088269

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014714, upper bound: 0.0015926
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014714, upper bound: 0.0015926
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000721, 0.0000747
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026995, 0.0027963
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032395, 0.0033557
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0238937, 0.0247508
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018824, 0.0018173
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019025, 0.0018367
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008933, 0.0009254
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0064144, 0.0061923
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050889, 0.0049126
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0091528, 0.0088359

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014714, upper bound: 0.0015926
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014714, upper bound: 0.0015926
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000735, 0.0000743
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027541, 0.0027808
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0033050, 0.0033370
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0243774, 0.0246134
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018720, 0.0018540
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018920, 0.0018738
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009114, 0.0009203
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0063788, 0.0063176
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050606, 0.0050121
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0091020, 0.0090148

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017810, upper bound: 0.0018734
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0017199, upper bound: 0.0019439
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000736, 0.0000742
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027553, 0.0027793
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0033065, 0.0033352
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0243883, 0.0246001
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018710, 0.0018549
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018910, 0.0018747
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009118, 0.0009198
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0063753, 0.0063204
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050579, 0.0050143
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0090971, 0.0090188

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013317, upper bound: 0.0013954
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013317, upper bound: 0.0013954
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000728, 0.0000745
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027264, 0.0027910
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032718, 0.0033493
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0241326, 0.0247038
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018789, 0.0018354
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018989, 0.0018550
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009023, 0.0009236
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0064022, 0.0062542
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050792, 0.0049618
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0091354, 0.0089242

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018337, upper bound: 0.0019787
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018186, upper bound: 0.0019955
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000728, 0.0000745
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027277, 0.0027895
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032734, 0.0033475
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0241437, 0.0246907
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018779, 0.0018363
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018979, 0.0018559
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009027, 0.0009231
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0063988, 0.0062571
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050765, 0.0049641
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0091306, 0.0089283

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018333, upper bound: 0.0019754
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018197, upper bound: 0.0019948
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000784, 0.0000768
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029342, 0.0028754
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035212, 0.0034506
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0259717, 0.0254507
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019357, 0.0019753
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019563, 0.0019964
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009710, 0.0009516
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065958, 0.0067308
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052328, 0.0053399
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094116, 0.0096043

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018156, upper bound: 0.0016989
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0018156, upper bound: 0.0016989
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000783, 0.0000769
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029316, 0.0028789
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035181, 0.0034548
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0259489, 0.0254816
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019380, 0.0019736
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019587, 0.0019946
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009702, 0.0009527
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0066038, 0.0067249
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052391, 0.0053352
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094231, 0.0095959

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014855
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014855
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000791, 0.0000766
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029604, 0.0028673
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035526, 0.0034408
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0262035, 0.0253789
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019302, 0.0019929
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019508, 0.0020142
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009797, 0.0009489
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065772, 0.0067909
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052180, 0.0053875
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093851, 0.0096900

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014841
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014841
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000790, 0.0000766
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0029578, 0.0028700
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0035495, 0.0034441
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0261807, 0.0254029
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019320, 0.0019912
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019527, 0.0020125
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009789, 0.0009498
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0065834, 0.0067850
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052229, 0.0053829
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0093940, 0.0096816

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014855
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014855
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000727, 0.0000715
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027212, 0.0026786
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032656, 0.0032144
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0240863, 0.0237089
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018032, 0.0018319
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018225, 0.0018515
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009005, 0.0008864
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0061444, 0.0062422
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0048746, 0.0049522
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0087675, 0.0089071

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017618, upper bound: 0.0015638
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016945, upper bound: 0.0016318
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000731, 0.0000709
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027386, 0.0026569
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032865, 0.0031883
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0242404, 0.0235166
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017886, 0.0018436
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018077, 0.0018633
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009063, 0.0008792
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0060945, 0.0062821
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0048351, 0.0049839
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0086964, 0.0089641

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015272, upper bound: 0.0014415
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015272, upper bound: 0.0014415
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000691, 0.0000675
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0025895, 0.0025294
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0031075, 0.0030354
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0229205, 0.0223887
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017028, 0.0017432
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0017210, 0.0017618
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008570, 0.0008371
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0058022, 0.0059400
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0046032, 0.0047125
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0082793, 0.0084760

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0012991
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0012991
time: 0.80 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0017919, upper bound: 0.0019373
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0017758, upper bound: 0.0019475
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0013751, upper bound: 0.0014333
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0013751, upper bound: 0.0014333
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0016576, upper bound: 0.0017363
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0015985, upper bound: 0.0017903
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0013838, upper bound: 0.0014698
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0013838, upper bound: 0.0014698
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0014714, upper bound: 0.0015926
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0014714, upper bound: 0.0015926
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0014714, upper bound: 0.0015926
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0014714, upper bound: 0.0015926
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0017810, upper bound: 0.0018734
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0017199, upper bound: 0.0019439
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0013317, upper bound: 0.0013954
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0013317, upper bound: 0.0013954
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0018337, upper bound: 0.0019787
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0018186, upper bound: 0.0019955
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0018333, upper bound: 0.0019754
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0018197, upper bound: 0.0019948
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0018156, upper bound: 0.0016989
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0018156, upper bound: 0.0016989
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014855
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014855
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014841
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014841
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014855
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0015489, upper bound: 0.0014855
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0017618, upper bound: 0.0015638
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0016945, upper bound: 0.0016318
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0015272, upper bound: 0.0014415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0015272, upper bound: 0.0014415
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0012991
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.13
Output dim: 2, lower bound: -0.0014075, upper bound: 0.0012991

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000747, 0.0000770
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027976, 0.0028839
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0033573, 0.0034608
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0247628, 0.0255266
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019414, 0.0018834
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019622, 0.0019035
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009258, 0.0009544
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0066154, 0.0064175
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052484, 0.0050913
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094397, 0.0091573

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016740, upper bound: 0.0018129
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016740, upper bound: 0.0018128
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000748, 0.0000769
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0028000, 0.0028810
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0033602, 0.0034573
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0247841, 0.0255004
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0019395, 0.0018850
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0019602, 0.0019051
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009266, 0.0009534
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0066087, 0.0064230
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0052430, 0.0050957
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0094300, 0.0091651

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016702, upper bound: 0.0017831
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016120, upper bound: 0.0018421
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000651, 0.0000648
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0024363, 0.0024285
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0029237, 0.0029143
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0215645, 0.0214953
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016348, 0.0016401
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016523, 0.0016576
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008063, 0.0008037
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0055707, 0.0055886
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044195, 0.0044337
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0079489, 0.0079745

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016102, upper bound: 0.0017009
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016102, upper bound: 0.0017009
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000642, 0.0000657
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0024030, 0.0024611
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0028837, 0.0029535
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0212697, 0.0217843
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016568, 0.0016177
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016745, 0.0016350
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0007952, 0.0008145
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0056456, 0.0055122
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0044789, 0.0043731
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0080558, 0.0078655

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015220, upper bound: 0.0017235
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015220, upper bound: 0.0017235
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000714, 0.0000732
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026733, 0.0027408
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032080, 0.0032891
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236620, 0.0242601
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018451, 0.0017996
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018648, 0.0018188
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008847, 0.0009070
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062872, 0.0061322
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049880, 0.0048650
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0089714, 0.0087502

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017294, upper bound: 0.0018153
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0016655, upper bound: 0.0018817
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000715, 0.0000731
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026763, 0.0027376
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032117, 0.0032853
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236889, 0.0242317
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018430, 0.0018017
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018626, 0.0018209
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008857, 0.0009060
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062799, 0.0061392
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049821, 0.0048705
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0089609, 0.0087601

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012744, upper bound: 0.0013557
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012744, upper bound: 0.0013557
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000714, 0.0000732
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026744, 0.0027394
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032094, 0.0032874
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236719, 0.0242471
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018441, 0.0018004
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018638, 0.0018196
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008851, 0.0009066
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062838, 0.0061348
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049853, 0.0048671
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0089665, 0.0087539

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016715, upper bound: 0.0018057
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0016715, upper bound: 0.0018057
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000715, 0.0000731
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026776, 0.0027363
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032132, 0.0032837
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0237001, 0.0242201
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018421, 0.0018025
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018618, 0.0018218
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008861, 0.0009056
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062769, 0.0061421
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049798, 0.0048728
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0089566, 0.0087643

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018110, upper bound: 0.0019916
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018162, upper bound: 0.0019839
time: 1.04 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0016740, upper bound: 0.0018129
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0016740, upper bound: 0.0018128
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0016702, upper bound: 0.0017831
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0016120, upper bound: 0.0018421
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0016102, upper bound: 0.0017009
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0016102, upper bound: 0.0017009
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0015220, upper bound: 0.0017235
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0015220, upper bound: 0.0017235
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0017294, upper bound: 0.0018153
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0016655, upper bound: 0.0018817
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0012744, upper bound: 0.0013557
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0012744, upper bound: 0.0013557
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0016715, upper bound: 0.0018057
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0016715, upper bound: 0.0018057
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0018110, upper bound: 0.0019916
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 2, lower bound: -0.0018162, upper bound: 0.0019839

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000618, 0.0000644
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0023131, 0.0024105
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0027758, 0.0028927
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0204736, 0.0213359
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016227, 0.0015571
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016400, 0.0015738
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0007655, 0.0007977
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0055294, 0.0053059
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0043868, 0.0042095
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0078900, 0.0075711

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014723, upper bound: 0.0016640
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014723, upper bound: 0.0016640
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000708, 0.0000725
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026526, 0.0027164
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0031832, 0.0032598
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0234789, 0.0240441
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018287, 0.0017857
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018482, 0.0018048
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008778, 0.0008990
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062312, 0.0060848
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049436, 0.0048274
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0088915, 0.0086825

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0017095, upper bound: 0.0018231
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0016400, upper bound: 0.0018921
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000710, 0.0000724
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026577, 0.0027114
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0031893, 0.0032538
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0235240, 0.0239997
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018253, 0.0017891
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018448, 0.0018082
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008795, 0.0008973
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062197, 0.0060964
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049344, 0.0048366
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0088751, 0.0086991

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012710, upper bound: 0.0013435
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012710, upper bound: 0.0013435
time: 0.91 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 2.54 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.54
Output dim: 2, lower bound: -0.0014723, upper bound: 0.0016640
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.54
Output dim: 2, lower bound: -0.0014723, upper bound: 0.0016640
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.54
Output dim: 2, lower bound: -0.0017095, upper bound: 0.0018231
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.54
Output dim: 2, lower bound: -0.0016400, upper bound: 0.0018921
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.54
Output dim: 2, lower bound: -0.0012710, upper bound: 0.0013435
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.54
Output dim: 2, lower bound: -0.0012710, upper bound: 0.0013435

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000613, 0.0000639
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0022957, 0.0023929
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0027550, 0.0028716
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0203202, 0.0211805
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0016109, 0.0015455
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0016281, 0.0015620
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0007597, 0.0007919
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0054891, 0.0052662
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0043548, 0.0041779
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0078325, 0.0075144

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015263, upper bound: 0.0017606
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0015263, upper bound: 0.0017606
time: 1.02 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 4.60 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 4.60
Output dim: 2, lower bound: -0.0015263, upper bound: 0.0017606
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 4.60
Output dim: 2, lower bound: -0.0015263, upper bound: 0.0017606

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.93 + 234.39 = 237.31 seconds
