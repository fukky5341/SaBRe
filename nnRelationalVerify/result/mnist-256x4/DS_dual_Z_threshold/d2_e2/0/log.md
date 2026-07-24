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
execution time: IAR + RelationalAnalysis = 1.86 + 2.11 = 3.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0022462, upper bound: 0.0022461

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022193, upper bound: 0.0022316
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0022316, upper bound: 0.0022193
time: 1.16 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.55 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.55
Output dim: 2, lower bound: -0.0022193, upper bound: 0.0022316
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.55
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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020500, upper bound: 0.0020536
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020506, upper bound: 0.0020536
time: 1.16 seconds

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

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020536, upper bound: 0.0020506
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0020536, upper bound: 0.0020500
time: 1.47 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.00 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 2, lower bound: -0.0020500, upper bound: 0.0020536
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 2, lower bound: -0.0020506, upper bound: 0.0020536
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 2, lower bound: -0.0020536, upper bound: 0.0020506
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.00
Output dim: 2, lower bound: -0.0020536, upper bound: 0.0020500

## BFS DS instance: DS_DSZ1_DSZ1

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

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019506, upper bound: 0.0018925
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018903, upper bound: 0.0019546
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019506, upper bound: 0.0018919
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018913, upper bound: 0.0019546
time: 1.40 seconds

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

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019547, upper bound: 0.0018913
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018919, upper bound: 0.0019506
time: 1.00 seconds

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

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019547, upper bound: 0.0018903
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018925, upper bound: 0.0019506
time: 1.39 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.98 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 2, lower bound: -0.0019506, upper bound: 0.0018925
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 2, lower bound: -0.0018903, upper bound: 0.0019546
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 2, lower bound: -0.0019506, upper bound: 0.0018919
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 2, lower bound: -0.0018913, upper bound: 0.0019546
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 2, lower bound: -0.0019547, upper bound: 0.0018913
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 2, lower bound: -0.0018919, upper bound: 0.0019506
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 2, lower bound: -0.0019547, upper bound: 0.0018903
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 2, lower bound: -0.0018925, upper bound: 0.0019506

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000726, 0.0000738
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027191, 0.0027651
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032631, 0.0033183
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0240677, 0.0244748
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018615, 0.0018305
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018813, 0.0018500
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008999, 0.0009151
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0063429, 0.0062374
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050321, 0.0049484
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0090508, 0.0089002

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019032, upper bound: 0.0018346
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018930, upper bound: 0.0018439
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000717, 0.0000745
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026862, 0.0027909
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032235, 0.0033492
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0237760, 0.0247032
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018788, 0.0018083
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018989, 0.0018276
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008889, 0.0009236
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0064021, 0.0061618
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050791, 0.0048884
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0091352, 0.0087923

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018415, upper bound: 0.0018964
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018315, upper bound: 0.0019069
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000737, 0.0000718
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027587, 0.0026870
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0033106, 0.0032245
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0244183, 0.0237836
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018089, 0.0018572
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018282, 0.0018770
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009130, 0.0008892
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0061637, 0.0063282
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0048900, 0.0050205
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0087951, 0.0090299

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019032, upper bound: 0.0018344
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018930, upper bound: 0.0018433
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000728, 0.0000726
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027258, 0.0027202
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032710, 0.0032644
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0241266, 0.0240774
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018312, 0.0018350
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018508, 0.0018546
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009021, 0.0009002
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062399, 0.0062526
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049504, 0.0049605
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0089038, 0.0089220

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018425, upper bound: 0.0018964
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018322, upper bound: 0.0019069
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000726, 0.0000738
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027202, 0.0027643
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032644, 0.0033173
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0240774, 0.0244675
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018609, 0.0018312
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018808, 0.0018508
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009002, 0.0009148
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0063410, 0.0062399
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050306, 0.0049504
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0090481, 0.0089038

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019069, upper bound: 0.0018322
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018964, upper bound: 0.0018425
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000718, 0.0000745
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026870, 0.0027900
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032245, 0.0033482
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0237836, 0.0246955
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018782, 0.0018089
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018983, 0.0018282
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008892, 0.0009233
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0064001, 0.0061637
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0050775, 0.0048900
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0091324, 0.0087951

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018433, upper bound: 0.0018930
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018344, upper bound: 0.0019032
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000737, 0.0000717
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027598, 0.0026862
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0033119, 0.0032235
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0244280, 0.0237760
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018083, 0.0018579
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018276, 0.0018777
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009133, 0.0008889
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0061618, 0.0063307
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0048884, 0.0050225
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0087923, 0.0090335

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0019069, upper bound: 0.0018315
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018964, upper bound: 0.0018415
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000728, 0.0000726
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027266, 0.0027191
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032721, 0.0032631
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0241341, 0.0240678
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018305, 0.0018355
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018500, 0.0018551
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0009023, 0.0008999
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062374, 0.0062546
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049484, 0.0049621
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0089002, 0.0089248

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 84

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018439, upper bound: 0.0018930
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0018346, upper bound: 0.0019031
time: 1.38 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.98 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0019032, upper bound: 0.0018346
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018930, upper bound: 0.0018439
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018415, upper bound: 0.0018964
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018315, upper bound: 0.0019069
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0019032, upper bound: 0.0018344
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018930, upper bound: 0.0018433
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018425, upper bound: 0.0018964
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018322, upper bound: 0.0019069
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0019069, upper bound: 0.0018322
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018964, upper bound: 0.0018425
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018433, upper bound: 0.0018930
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018344, upper bound: 0.0019032
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0019069, upper bound: 0.0018315
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018964, upper bound: 0.0018415
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018439, upper bound: 0.0018930
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.98
Output dim: 2, lower bound: -0.0018346, upper bound: 0.0019031

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000712, 0.0000725
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026675, 0.0027138
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032011, 0.0032567
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236107, 0.0240210
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018269, 0.0017957
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018464, 0.0018149
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008828, 0.0008981
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062253, 0.0061189
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049388, 0.0048545
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0088830, 0.0087312

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000712, 0.0000724
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026678, 0.0027111
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032015, 0.0032534
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236139, 0.0239966
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018251, 0.0017960
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018446, 0.0018152
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008829, 0.0008972
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062189, 0.0061198
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049338, 0.0048551
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0088739, 0.0087324

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000703, 0.0000732
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026318, 0.0027396
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0031583, 0.0032877
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0232953, 0.0242494
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018443, 0.0017717
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018640, 0.0017907
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008710, 0.0009066
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062844, 0.0060372
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049858, 0.0047896
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0089674, 0.0086146

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000704, 0.0000731
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026349, 0.0027380
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0031620, 0.0032857
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0233222, 0.0242345
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018432, 0.0017738
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018629, 0.0017927
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008720, 0.0009061
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062806, 0.0060442
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049827, 0.0047951
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0089619, 0.0086245

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012019
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012019
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000723, 0.0000704
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027059, 0.0026357
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032472, 0.0031630
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0239507, 0.0233297
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017744, 0.0018216
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0017933, 0.0018410
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008955, 0.0008723
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0060461, 0.0062070
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0047967, 0.0049244
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0086273, 0.0088570

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000723, 0.0000703
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027063, 0.0026327
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032476, 0.0031593
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0239540, 0.0233024
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017723, 0.0018218
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0017912, 0.0018413
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008956, 0.0008712
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0060390, 0.0062079
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0047911, 0.0049250
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0086172, 0.0088582

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000713, 0.0000713
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026703, 0.0026689
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032044, 0.0032028
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236353, 0.0236236
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017967, 0.0017976
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018159, 0.0018168
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008837, 0.0008833
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0061223, 0.0061253
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0048571, 0.0048595
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0087360, 0.0087403

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000714, 0.0000713
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026733, 0.0026684
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032081, 0.0032022
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236622, 0.0236188
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017963, 0.0017996
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018155, 0.0018189
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008847, 0.0008831
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0061210, 0.0061323
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0048561, 0.0048651
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0087342, 0.0087503

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012019
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012019
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000713, 0.0000724
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026684, 0.0027130
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032022, 0.0032557
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236188, 0.0240137
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018264, 0.0017963
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018459, 0.0018155
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008831, 0.0008978
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062234, 0.0061210
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049373, 0.0048561
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0088803, 0.0087342

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000713, 0.0000724
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026689, 0.0027102
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032028, 0.0032524
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236236, 0.0239892
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018245, 0.0017967
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018440, 0.0018159
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008833, 0.0008969
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062170, 0.0061223
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049323, 0.0048571
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0088712, 0.0087360

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000703, 0.0000731
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026327, 0.0027388
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0031593, 0.0032867
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0233024, 0.0242417
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018437, 0.0017723
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018634, 0.0017912
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008712, 0.0009064
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062825, 0.0060390
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049842, 0.0047911
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0089646, 0.0086172

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000704, 0.0000731
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026357, 0.0027372
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0031630, 0.0032848
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0233297, 0.0242279
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0018427, 0.0017744
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018624, 0.0017933
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008723, 0.0009058
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0062789, 0.0060461
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0049814, 0.0047967
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0089595, 0.0086273

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012018
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012018
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000723, 0.0000704
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027068, 0.0026349
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032483, 0.0031620
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0239588, 0.0233222
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017738, 0.0018222
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0017927, 0.0018417
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008958, 0.0008720
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0060442, 0.0062091
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0047951, 0.0049260
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0086245, 0.0088600

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000723, 0.0000703
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0027074, 0.0026318
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032489, 0.0031583
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0239637, 0.0232953
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017717, 0.0018226
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0017907, 0.0018420
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008960, 0.0008710
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0060372, 0.0062104
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0047896, 0.0049270
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0086146, 0.0088617

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000713, 0.0000712
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026711, 0.0026678
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032054, 0.0032015
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236425, 0.0236139
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017960, 0.0017981
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018152, 0.0018173
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008840, 0.0008829
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0061198, 0.0061271
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0048551, 0.0048610
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0087324, 0.0087430

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0041910, -0.0041119, -0.0041910, -0.0041119, -0.0000714, 0.0000712
1: -0.0096089, -0.0066455, -0.0096089, -0.0066455, -0.0026742, 0.0026675
2: 0.9649323, 0.9684886, 0.9649323, 0.9684886, -0.0032091, 0.0032011
3: -0.0123469, 0.0138834, -0.0123469, 0.0138834, -0.0236698, 0.0236107
4: -0.0017489, 0.0002460, -0.0017489, 0.0002460, -0.0017957, 0.0018002
5: 0.0155027, 0.0175190, 0.0155027, 0.0175190, -0.0018149, 0.0018194
6: 0.0033856, 0.0043663, 0.0033856, 0.0043663, -0.0008850, 0.0008828
7: -0.0113762, -0.0045784, -0.0113762, -0.0045784, -0.0061189, 0.0061342
8: 0.0077038, 0.0130968, 0.0077038, 0.0130968, -0.0048545, 0.0048666
9: 0.0115806, 0.0212805, 0.0115806, 0.0212805, -0.0087312, 0.0087531

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012018
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012018
time: 1.20 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.39 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012019
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012019
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012019
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012019
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012018
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012018
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012019, upper bound: 0.0011859
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0012004, upper bound: 0.0011870
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011870, upper bound: 0.0012004
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012018
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.39
Output dim: 2, lower bound: -0.0011859, upper bound: 0.0012018

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.97 + 115.65 = 119.62 seconds
