## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06738792


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099)
1: (-0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061)
2: (-0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713)
3: (-0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883)
4: (-0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926)
5: (-0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117)
6: (-0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370)
7: (-0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255)
8: (-0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314)
9: (0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.76 + 7.55 = 8.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0842349, upper bound: 0.0842349

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0841254, upper bound: 0.0838611
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0838611, upper bound: 0.0841254
time: 4.74 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.42 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.42
Output dim: 9, lower bound: -0.0841254, upper bound: 0.0838611
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.42
Output dim: 9, lower bound: -0.0838611, upper bound: 0.0841254

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0823884, upper bound: 0.0819500
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0820922, upper bound: 0.0821538
time: 1.80 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0747519, upper bound: 0.0750037
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0747519, upper bound: 0.0750037
time: 1.56 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.93 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 9, lower bound: -0.0823884, upper bound: 0.0819500
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 9, lower bound: -0.0820922, upper bound: 0.0821538
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 9, lower bound: -0.0747519, upper bound: 0.0750037
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 9, lower bound: -0.0747519, upper bound: 0.0750037

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0821805, upper bound: 0.0816057
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0819846, upper bound: 0.0817427
time: 3.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0775662, upper bound: 0.0776062
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0775662, upper bound: 0.0776062
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0747519, upper bound: 0.0749744
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0747329, upper bound: 0.0750037
time: 3.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725834, upper bound: 0.0726382
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725124, upper bound: 0.0727591
time: 1.52 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.43 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 9, lower bound: -0.0821805, upper bound: 0.0816057
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 9, lower bound: -0.0819846, upper bound: 0.0817427
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 9, lower bound: -0.0775662, upper bound: 0.0776062
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 9, lower bound: -0.0775662, upper bound: 0.0776062
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 9, lower bound: -0.0747519, upper bound: 0.0749744
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 9, lower bound: -0.0747329, upper bound: 0.0750037
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 9, lower bound: -0.0725834, upper bound: 0.0726382
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 9, lower bound: -0.0725124, upper bound: 0.0727591

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818613, upper bound: 0.0812907
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818602, upper bound: 0.0812974
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0819846, upper bound: 0.0817086
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0819160, upper bound: 0.0817427
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708492, upper bound: 0.0708131
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708492, upper bound: 0.0708131
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0775662, upper bound: 0.0774961
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0774984, upper bound: 0.0776059
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0715245, upper bound: 0.0716358
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0715245, upper bound: 0.0716358
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0743399, upper bound: 0.0746380
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0743664, upper bound: 0.0745962
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0723867, upper bound: 0.0723676
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0722921, upper bound: 0.0724408
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0719785, upper bound: 0.0722067
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0719589, upper bound: 0.0722218
time: 1.72 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.80 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0818613, upper bound: 0.0812907
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0818602, upper bound: 0.0812974
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0819846, upper bound: 0.0817086
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0819160, upper bound: 0.0817427
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0708492, upper bound: 0.0708131
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0708492, upper bound: 0.0708131
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0775662, upper bound: 0.0774961
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0774984, upper bound: 0.0776059
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0715245, upper bound: 0.0716358
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0715245, upper bound: 0.0716358
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0743399, upper bound: 0.0746380
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0743664, upper bound: 0.0745962
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0723867, upper bound: 0.0723676
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0722921, upper bound: 0.0724408
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0719785, upper bound: 0.0722067
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 9, lower bound: -0.0719589, upper bound: 0.0722218

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0743136, upper bound: 0.0739260
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0743136, upper bound: 0.0739260
time: 2.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0805866, upper bound: 0.0800636
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0805866, upper bound: 0.0800636
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772892, upper bound: 0.0771381
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772892, upper bound: 0.0771381
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0801466, upper bound: 0.0800172
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0801353, upper bound: 0.0800257
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0587433, upper bound: 0.0586951
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0587433, upper bound: 0.0586951
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708240, upper bound: 0.0706867
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0707280, upper bound: 0.0707882
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0693851, upper bound: 0.0693233
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0693851, upper bound: 0.0693233
time: 2.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0707280, upper bound: 0.0707882
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0707280, upper bound: 0.0707882
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0689471, upper bound: 0.0690047
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0689471, upper bound: 0.0690047
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0710061, upper bound: 0.0711124
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0709915, upper bound: 0.0711232
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0736256, upper bound: 0.0739031
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0736062, upper bound: 0.0739133
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0610742, upper bound: 0.0611365
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0610742, upper bound: 0.0611365
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0719868, upper bound: 0.0719984
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0720179, upper bound: 0.0719542
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0722913, upper bound: 0.0724192
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0722576, upper bound: 0.0724408
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717810, upper bound: 0.0718800
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716972, upper bound: 0.0720074
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0719589, upper bound: 0.0721990
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0719452, upper bound: 0.0722218
time: 1.34 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.28 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0743136, upper bound: 0.0739260
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0743136, upper bound: 0.0739260
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0805866, upper bound: 0.0800636
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0805866, upper bound: 0.0800636
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0772892, upper bound: 0.0771381
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0772892, upper bound: 0.0771381
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0801466, upper bound: 0.0800172
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0801353, upper bound: 0.0800257
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0587433, upper bound: 0.0586951
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0587433, upper bound: 0.0586951
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0708240, upper bound: 0.0706867
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0707280, upper bound: 0.0707882
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0693851, upper bound: 0.0693233
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0693851, upper bound: 0.0693233
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0707280, upper bound: 0.0707882
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0707280, upper bound: 0.0707882
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0689471, upper bound: 0.0690047
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0689471, upper bound: 0.0690047
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0710061, upper bound: 0.0711124
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0709915, upper bound: 0.0711232
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0736256, upper bound: 0.0739031
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0736062, upper bound: 0.0739133
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0610742, upper bound: 0.0611365
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0610742, upper bound: 0.0611365
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0719868, upper bound: 0.0719984
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0720179, upper bound: 0.0719542
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0722913, upper bound: 0.0724192
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0722576, upper bound: 0.0724408
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0717810, upper bound: 0.0718800
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0716972, upper bound: 0.0720074
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0719589, upper bound: 0.0721990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 9, lower bound: -0.0719452, upper bound: 0.0722218

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0598849, upper bound: 0.0598214
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0598849, upper bound: 0.0598214
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705530, upper bound: 0.0702666
time: 6.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705530, upper bound: 0.0702666
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0756337, upper bound: 0.0752605
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0756337, upper bound: 0.0752605
time: 3.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0779291, upper bound: 0.0779734
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0784042, upper bound: 0.0774569
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0770955, upper bound: 0.0769510
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0771006, upper bound: 0.0769512
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769637, upper bound: 0.0763847
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0764580, upper bound: 0.0767742
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773256, upper bound: 0.0777405
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0778572, upper bound: 0.0772547
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798135, upper bound: 0.0796929
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798135, upper bound: 0.0796974
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0704448, upper bound: 0.0703736
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705127, upper bound: 0.0703347
time: 2.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0587131, upper bound: 0.0586850
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0587131, upper bound: 0.0586850
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0691807, upper bound: 0.0689709
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690719, upper bound: 0.0691092
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0691807, upper bound: 0.0689709
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690719, upper bound: 0.0691092
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0674466, upper bound: 0.0678311
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677386, upper bound: 0.0674815
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680100, upper bound: 0.0680758
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680100, upper bound: 0.0680758
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685837, upper bound: 0.0686423
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685704, upper bound: 0.0686507
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0651260, upper bound: 0.0651085
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0651184, upper bound: 0.0651233
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685837, upper bound: 0.0686423
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685837, upper bound: 0.0686423
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705918, upper bound: 0.0707576
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706251, upper bound: 0.0707076
time: 2.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716129, upper bound: 0.0717250
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0715555, upper bound: 0.0718482
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705560, upper bound: 0.0707892
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705560, upper bound: 0.0707893
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716891, upper bound: 0.0716920
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716791, upper bound: 0.0716913
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717156, upper bound: 0.0716564
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717108, upper bound: 0.0716573
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690327, upper bound: 0.0691384
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690327, upper bound: 0.0691384
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0599787, upper bound: 0.0600728
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0599787, upper bound: 0.0600728
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717810, upper bound: 0.0718455
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717670, upper bound: 0.0718800
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0713174, upper bound: 0.0711572
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0709051, upper bound: 0.0716363
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0715736, upper bound: 0.0713631
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0711714, upper bound: 0.0718136
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0688232, upper bound: 0.0690259
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0688232, upper bound: 0.0690259
time: 1.88 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.08 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0598849, upper bound: 0.0598214
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0598849, upper bound: 0.0598214
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0705530, upper bound: 0.0702666
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0705530, upper bound: 0.0702666
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0756337, upper bound: 0.0752605
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0756337, upper bound: 0.0752605
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0779291, upper bound: 0.0779734
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0784042, upper bound: 0.0774569
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0770955, upper bound: 0.0769510
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0771006, upper bound: 0.0769512
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0769637, upper bound: 0.0763847
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0764580, upper bound: 0.0767742
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0773256, upper bound: 0.0777405
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0778572, upper bound: 0.0772547
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0798135, upper bound: 0.0796929
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0798135, upper bound: 0.0796974
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0704448, upper bound: 0.0703736
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0705127, upper bound: 0.0703347
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0587131, upper bound: 0.0586850
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0587131, upper bound: 0.0586850
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0691807, upper bound: 0.0689709
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0690719, upper bound: 0.0691092
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0691807, upper bound: 0.0689709
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0690719, upper bound: 0.0691092
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0674466, upper bound: 0.0678311
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0677386, upper bound: 0.0674815
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0680100, upper bound: 0.0680758
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0680100, upper bound: 0.0680758
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0685837, upper bound: 0.0686423
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0685704, upper bound: 0.0686507
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0651260, upper bound: 0.0651085
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0651184, upper bound: 0.0651233
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0685837, upper bound: 0.0686423
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0685837, upper bound: 0.0686423
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0705918, upper bound: 0.0707576
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0706251, upper bound: 0.0707076
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0716129, upper bound: 0.0717250
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0715555, upper bound: 0.0718482
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0705560, upper bound: 0.0707892
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0705560, upper bound: 0.0707893
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0716891, upper bound: 0.0716920
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0716791, upper bound: 0.0716913
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0717156, upper bound: 0.0716564
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0717108, upper bound: 0.0716573
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0690327, upper bound: 0.0691384
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0690327, upper bound: 0.0691384
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0599787, upper bound: 0.0600728
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0599787, upper bound: 0.0600728
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0717810, upper bound: 0.0718455
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0717670, upper bound: 0.0718800
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0713174, upper bound: 0.0711572
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0709051, upper bound: 0.0716363
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0715736, upper bound: 0.0713631
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0711714, upper bound: 0.0718136
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0688232, upper bound: 0.0690259
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 9, lower bound: -0.0688232, upper bound: 0.0690259

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701865, upper bound: 0.0699497
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702376, upper bound: 0.0698994
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702194, upper bound: 0.0695906
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0698516, upper bound: 0.0699273
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751978, upper bound: 0.0748838
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0752644, upper bound: 0.0748517
time: 2.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0756337, upper bound: 0.0752168
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0755504, upper bound: 0.0752605
time: 2.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0388280, 0.0389511
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0775430, upper bound: 0.0776261
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0775823, upper bound: 0.0776130
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0386022, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0693537, upper bound: 0.0688279
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0693537, upper bound: 0.0688279
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0704204, upper bound: 0.0702473
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0704204, upper bound: 0.0702473
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0754538, upper bound: 0.0753376
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0754538, upper bound: 0.0753376
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0687597, upper bound: 0.0683373
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0687597, upper bound: 0.0683373
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0683661, upper bound: 0.0686908
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0683661, upper bound: 0.0686908
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0386937, 0.0388962
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0771473, upper bound: 0.0774950
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0771473, upper bound: 0.0774961
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0384503, 0.0390867
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740188, upper bound: 0.0735279
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740188, upper bound: 0.0735279
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0714981, upper bound: 0.0715631
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0714981, upper bound: 0.0715631
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786035, upper bound: 0.0784820
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786035, upper bound: 0.0784820
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677346, upper bound: 0.0676564
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677346, upper bound: 0.0676564
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0703559, upper bound: 0.0701654
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0703559, upper bound: 0.0701758
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0687510, upper bound: 0.0686260
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0688414, upper bound: 0.0685594
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0583767, upper bound: 0.0583727
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0583767, upper bound: 0.0583727
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0687510, upper bound: 0.0686260
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0688414, upper bound: 0.0685594
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0648684, upper bound: 0.0648518
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0648684, upper bound: 0.0648518
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0390426, 0.0386895
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0657438, upper bound: 0.0660682
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0657438, upper bound: 0.0660682
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0388165, 0.0389001
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0673698, upper bound: 0.0671706
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0674200, upper bound: 0.0671094
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0676422, upper bound: 0.0677413
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0676797, upper bound: 0.0677245
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0678829, upper bound: 0.0679460
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0678817, upper bound: 0.0679478
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0681859, upper bound: 0.0678751
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677912, upper bound: 0.0682294
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0683397, upper bound: 0.0682885
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682192, upper bound: 0.0684145
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0683048, upper bound: 0.0683693
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0683051, upper bound: 0.0683580
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682005, upper bound: 0.0682877
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682288, upper bound: 0.0682215
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0703711, upper bound: 0.0704277
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702659, upper bound: 0.0705360
time: 2.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0704032, upper bound: 0.0703932
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702855, upper bound: 0.0704898
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0714140, upper bound: 0.0714172
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0713134, upper bound: 0.0715240
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680381, upper bound: 0.0682192
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680381, upper bound: 0.0682192
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680483, upper bound: 0.0681403
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680124, upper bound: 0.0682274
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702962, upper bound: 0.0705338
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702941, upper bound: 0.0705215
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0713202, upper bound: 0.0708782
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0707994, upper bound: 0.0713172
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0713125, upper bound: 0.0708825
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0707860, upper bound: 0.0713154
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0595365, upper bound: 0.0595268
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0595365, upper bound: 0.0595268
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680175, upper bound: 0.0679695
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680175, upper bound: 0.0679695
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0659206, upper bound: 0.0665183
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0664307, upper bound: 0.0661104
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0659199, upper bound: 0.0665183
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0664307, upper bound: 0.0661116
time: 2.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0686787, upper bound: 0.0685892
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0686787, upper bound: 0.0685892
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0714766, upper bound: 0.0715878
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0714713, upper bound: 0.0715831
time: 1.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0681806, upper bound: 0.0680215
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0681806, upper bound: 0.0680215
time: 2.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0590190, upper bound: 0.0592092
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0590190, upper bound: 0.0592092
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0684332, upper bound: 0.0686095
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0688382, upper bound: 0.0682911
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0592979, upper bound: 0.0594316
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0592979, upper bound: 0.0594316
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0684200, upper bound: 0.0682436
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680887, upper bound: 0.0686369
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685102, upper bound: 0.0687093
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685098, upper bound: 0.0687036
time: 1.69 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.94 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0701865, upper bound: 0.0699497
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0702376, upper bound: 0.0698994
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0702194, upper bound: 0.0695906
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0698516, upper bound: 0.0699273
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0751978, upper bound: 0.0748838
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0752644, upper bound: 0.0748517
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0756337, upper bound: 0.0752168
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0755504, upper bound: 0.0752605
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0775430, upper bound: 0.0776261
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0775823, upper bound: 0.0776130
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0693537, upper bound: 0.0688279
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0693537, upper bound: 0.0688279
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0704204, upper bound: 0.0702473
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0704204, upper bound: 0.0702473
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0754538, upper bound: 0.0753376
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0754538, upper bound: 0.0753376
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0687597, upper bound: 0.0683373
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0687597, upper bound: 0.0683373
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0683661, upper bound: 0.0686908
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0683661, upper bound: 0.0686908
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0771473, upper bound: 0.0774950
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0771473, upper bound: 0.0774961
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0740188, upper bound: 0.0735279
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0740188, upper bound: 0.0735279
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0714981, upper bound: 0.0715631
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0714981, upper bound: 0.0715631
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0786035, upper bound: 0.0784820
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0786035, upper bound: 0.0784820
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0677346, upper bound: 0.0676564
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0677346, upper bound: 0.0676564
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0703559, upper bound: 0.0701654
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0703559, upper bound: 0.0701758
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0687510, upper bound: 0.0686260
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0688414, upper bound: 0.0685594
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0583767, upper bound: 0.0583727
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0583767, upper bound: 0.0583727
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0687510, upper bound: 0.0686260
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0688414, upper bound: 0.0685594
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0648684, upper bound: 0.0648518
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0648684, upper bound: 0.0648518
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0657438, upper bound: 0.0660682
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0657438, upper bound: 0.0660682
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0673698, upper bound: 0.0671706
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0674200, upper bound: 0.0671094
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0676422, upper bound: 0.0677413
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0676797, upper bound: 0.0677245
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0678829, upper bound: 0.0679460
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0678817, upper bound: 0.0679478
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0681859, upper bound: 0.0678751
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0677912, upper bound: 0.0682294
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0683397, upper bound: 0.0682885
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0682192, upper bound: 0.0684145
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0683048, upper bound: 0.0683693
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0683051, upper bound: 0.0683580
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0682005, upper bound: 0.0682877
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0682288, upper bound: 0.0682215
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0703711, upper bound: 0.0704277
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0702659, upper bound: 0.0705360
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0704032, upper bound: 0.0703932
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0702855, upper bound: 0.0704898
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0714140, upper bound: 0.0714172
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0713134, upper bound: 0.0715240
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0680381, upper bound: 0.0682192
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0680381, upper bound: 0.0682192
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0680483, upper bound: 0.0681403
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0680124, upper bound: 0.0682274
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0702962, upper bound: 0.0705338
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0702941, upper bound: 0.0705215
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0713202, upper bound: 0.0708782
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0707994, upper bound: 0.0713172
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0713125, upper bound: 0.0708825
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0707860, upper bound: 0.0713154
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0595365, upper bound: 0.0595268
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0595365, upper bound: 0.0595268
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0680175, upper bound: 0.0679695
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0680175, upper bound: 0.0679695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0659206, upper bound: 0.0665183
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0664307, upper bound: 0.0661104
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0659199, upper bound: 0.0665183
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0664307, upper bound: 0.0661116
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0686787, upper bound: 0.0685892
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0686787, upper bound: 0.0685892
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0714766, upper bound: 0.0715878
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0714713, upper bound: 0.0715831
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0681806, upper bound: 0.0680215
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0681806, upper bound: 0.0680215
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0590190, upper bound: 0.0592092
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0590190, upper bound: 0.0592092
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0684332, upper bound: 0.0686095
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0688382, upper bound: 0.0682911
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0592979, upper bound: 0.0594316
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0592979, upper bound: 0.0594316
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0684200, upper bound: 0.0682436
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0680887, upper bound: 0.0686369
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0685102, upper bound: 0.0687093
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 9, lower bound: -0.0685098, upper bound: 0.0687036

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701629, upper bound: 0.0698323
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700409, upper bound: 0.0699239
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0579136, upper bound: 0.0578330
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0579136, upper bound: 0.0578330
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685287, upper bound: 0.0678869
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685287, upper bound: 0.0678869
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0698233, upper bound: 0.0697969
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0698031, upper bound: 0.0699027
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0724800, upper bound: 0.0728113
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0730174, upper bound: 0.0721947
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725528, upper bound: 0.0727870
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0730592, upper bound: 0.0721645
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0729205, upper bound: 0.0731353
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0734344, upper bound: 0.0725149
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0676865, upper bound: 0.0676538
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0676865, upper bound: 0.0676538
time: 1.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0386700, 0.0387783
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0649620, upper bound: 0.0651705
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0649620, upper bound: 0.0651705
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0386552, 0.0387951
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0649990, upper bound: 0.0651486
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0649990, upper bound: 0.0651486
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0384927, 0.0390608
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0693537, upper bound: 0.0687575
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0692407, upper bound: 0.0688267
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0385084, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665211, upper bound: 0.0661416
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665211, upper bound: 0.0661416
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0686792, upper bound: 0.0685700
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0686792, upper bound: 0.0685700
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677425, upper bound: 0.0676286
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677425, upper bound: 0.0676286
time: 2.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0750901, upper bound: 0.0745813
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0746575, upper bound: 0.0749504
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0727496, upper bound: 0.0732007
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0733229, upper bound: 0.0726475
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0683416, upper bound: 0.0679991
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0684174, upper bound: 0.0679254
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0656504, upper bound: 0.0657302
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0661361, upper bound: 0.0653452
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0641778, upper bound: 0.0644392
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0641778, upper bound: 0.0644392
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680779, upper bound: 0.0683970
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680786, upper bound: 0.0683970
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0386821, 0.0389193
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0767160, upper bound: 0.0771568
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0767932, upper bound: 0.0770904
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0387192, 0.0388846
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0758283, upper bound: 0.0761992
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0758283, upper bound: 0.0761992
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0383667, 0.0390320
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0735840, upper bound: 0.0731997
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0736931, upper bound: 0.0730772
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0383957, 0.0390215
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0659338, upper bound: 0.0655784
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0659338, upper bound: 0.0655784
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0696154, upper bound: 0.0696583
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0696154, upper bound: 0.0696583
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680599, upper bound: 0.0683992
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0683537, upper bound: 0.0681315
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0679820, upper bound: 0.0680161
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0679820, upper bound: 0.0680161
time: 2.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0743468, upper bound: 0.0743526
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0743468, upper bound: 0.0743526
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0675172, upper bound: 0.0673750
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0674688, upper bound: 0.0674385
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0676066, upper bound: 0.0675232
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0676035, upper bound: 0.0675307
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0686876, upper bound: 0.0684944
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0686876, upper bound: 0.0684944
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0686866, upper bound: 0.0685095
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0686866, upper bound: 0.0685095
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0683788, upper bound: 0.0678979
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680068, upper bound: 0.0682431
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0684036, upper bound: 0.0681049
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0683775, upper bound: 0.0681544
time: 2.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0645288, upper bound: 0.0644343
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0645288, upper bound: 0.0644343
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0684547, upper bound: 0.0678408
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680957, upper bound: 0.0681780
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0387044, 0.0388257
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0671760, upper bound: 0.0667588
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0670751, upper bound: 0.0668666
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0643890, upper bound: 0.0647522
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0646576, upper bound: 0.0644760
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0675531, upper bound: 0.0675929
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0675521, upper bound: 0.0675959
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0675433, upper bound: 0.0672981
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673233, upper bound: 0.0676132
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0675116, upper bound: 0.0676134
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0675521, upper bound: 0.0675959
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0678055, upper bound: 0.0675187
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0678302, upper bound: 0.0674628
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0648009, upper bound: 0.0655361
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0650954, upper bound: 0.0651925
time: 1.44 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0701629, upper bound: 0.0698323
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0700409, upper bound: 0.0699239
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0579136, upper bound: 0.0578330
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0579136, upper bound: 0.0578330
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0685287, upper bound: 0.0678869
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0685287, upper bound: 0.0678869
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0698233, upper bound: 0.0697969
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0698031, upper bound: 0.0699027
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0724800, upper bound: 0.0728113
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0730174, upper bound: 0.0721947
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0725528, upper bound: 0.0727870
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0730592, upper bound: 0.0721645
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0729205, upper bound: 0.0731353
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0734344, upper bound: 0.0725149
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0676865, upper bound: 0.0676538
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0676865, upper bound: 0.0676538
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0649620, upper bound: 0.0651705
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0649620, upper bound: 0.0651705
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0649990, upper bound: 0.0651486
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0649990, upper bound: 0.0651486
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0693537, upper bound: 0.0687575
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0692407, upper bound: 0.0688267
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0665211, upper bound: 0.0661416
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0665211, upper bound: 0.0661416
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0686792, upper bound: 0.0685700
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0686792, upper bound: 0.0685700
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0677425, upper bound: 0.0676286
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0677425, upper bound: 0.0676286
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0750901, upper bound: 0.0745813
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0746575, upper bound: 0.0749504
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0727496, upper bound: 0.0732007
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0733229, upper bound: 0.0726475
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0683416, upper bound: 0.0679991
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0684174, upper bound: 0.0679254
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0656504, upper bound: 0.0657302
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0661361, upper bound: 0.0653452
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0641778, upper bound: 0.0644392
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0641778, upper bound: 0.0644392
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0680779, upper bound: 0.0683970
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0680786, upper bound: 0.0683970
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0767160, upper bound: 0.0771568
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0767932, upper bound: 0.0770904
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0758283, upper bound: 0.0761992
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0758283, upper bound: 0.0761992
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0735840, upper bound: 0.0731997
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0736931, upper bound: 0.0730772
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0659338, upper bound: 0.0655784
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0659338, upper bound: 0.0655784
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0696154, upper bound: 0.0696583
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0696154, upper bound: 0.0696583
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0680599, upper bound: 0.0683992
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0683537, upper bound: 0.0681315
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0679820, upper bound: 0.0680161
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0679820, upper bound: 0.0680161
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0743468, upper bound: 0.0743526
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0743468, upper bound: 0.0743526
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0675172, upper bound: 0.0673750
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0674688, upper bound: 0.0674385
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0676066, upper bound: 0.0675232
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0676035, upper bound: 0.0675307
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0686876, upper bound: 0.0684944
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0686876, upper bound: 0.0684944
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0686866, upper bound: 0.0685095
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0686866, upper bound: 0.0685095
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0683788, upper bound: 0.0678979
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0680068, upper bound: 0.0682431
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0684036, upper bound: 0.0681049
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0683775, upper bound: 0.0681544
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0645288, upper bound: 0.0644343
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0645288, upper bound: 0.0644343
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0684547, upper bound: 0.0678408
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0680957, upper bound: 0.0681780
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0671760, upper bound: 0.0667588
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0670751, upper bound: 0.0668666
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0643890, upper bound: 0.0647522
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0646576, upper bound: 0.0644760
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0675531, upper bound: 0.0675929
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0675521, upper bound: 0.0675959
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0675433, upper bound: 0.0672981
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0673233, upper bound: 0.0676132
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0675116, upper bound: 0.0676134
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0675521, upper bound: 0.0675959
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0678055, upper bound: 0.0675187
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0678302, upper bound: 0.0674628
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0648009, upper bound: 0.0655361
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 9, lower bound: -0.0650954, upper bound: 0.0651925
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0683397, upper bound: 0.0682885
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0682192, upper bound: 0.0684145
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0683048, upper bound: 0.0683693
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0683051, upper bound: 0.0683580
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0682005, upper bound: 0.0682877
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0682288, upper bound: 0.0682215
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0703711, upper bound: 0.0704277
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0702659, upper bound: 0.0705360
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0704032, upper bound: 0.0703932
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0702855, upper bound: 0.0704898
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0714140, upper bound: 0.0714172
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0713134, upper bound: 0.0715240
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0680381, upper bound: 0.0682192
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0680381, upper bound: 0.0682192
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0680483, upper bound: 0.0681403
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0680124, upper bound: 0.0682274
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0702962, upper bound: 0.0705338
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0702941, upper bound: 0.0705215
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0713202, upper bound: 0.0708782
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0707994, upper bound: 0.0713172
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0713125, upper bound: 0.0708825
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0707860, upper bound: 0.0713154
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0680175, upper bound: 0.0679695
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0680175, upper bound: 0.0679695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0686787, upper bound: 0.0685892
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0686787, upper bound: 0.0685892
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0714766, upper bound: 0.0715878
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0714713, upper bound: 0.0715831
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0681806, upper bound: 0.0680215
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0681806, upper bound: 0.0680215
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0684332, upper bound: 0.0686095
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0688382, upper bound: 0.0682911
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0684200, upper bound: 0.0682436
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0680887, upper bound: 0.0686369
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0685102, upper bound: 0.0687093
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 9, lower bound: -0.0685098, upper bound: 0.0687036

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 8.31 + 592.59 = 600.91 seconds
