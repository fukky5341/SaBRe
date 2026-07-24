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
execution time: IAR + RelationalAnalysis = 1.83 + 7.96 = 9.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0842349, upper bound: 0.0842349

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751305, upper bound: 0.0751305
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751305, upper bound: 0.0751305
time: 1.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.04 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.04
Output dim: 9, lower bound: -0.0751305, upper bound: 0.0751305
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.04
Output dim: 9, lower bound: -0.0751305, upper bound: 0.0751305

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

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728817, upper bound: 0.0727647
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0727647, upper bound: 0.0728817
time: 1.35 seconds

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

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728817, upper bound: 0.0727647
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0727647, upper bound: 0.0728817
time: 1.34 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.35 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 9, lower bound: -0.0728817, upper bound: 0.0727647
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 9, lower bound: -0.0727647, upper bound: 0.0728817
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 9, lower bound: -0.0728817, upper bound: 0.0727647
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.35
Output dim: 9, lower bound: -0.0727647, upper bound: 0.0728817

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

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728817, upper bound: 0.0727453
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728594, upper bound: 0.0727647
time: 1.34 seconds

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

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0727647, upper bound: 0.0728594
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0727453, upper bound: 0.0728817
time: 1.51 seconds

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

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728817, upper bound: 0.0727453
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728594, upper bound: 0.0727647
time: 1.35 seconds

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

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0727647, upper bound: 0.0728594
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0727453, upper bound: 0.0728817
time: 1.45 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.67 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0728817, upper bound: 0.0727453
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0728594, upper bound: 0.0727647
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0727647, upper bound: 0.0728594
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0727453, upper bound: 0.0728817
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0728817, upper bound: 0.0727453
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0728594, upper bound: 0.0727647
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0727647, upper bound: 0.0728594
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0727453, upper bound: 0.0728817

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

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0698457, upper bound: 0.0700966
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702543, upper bound: 0.0697266
time: 1.66 seconds

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

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0698265, upper bound: 0.0701166
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702274, upper bound: 0.0697385
time: 1.59 seconds

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

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0697385, upper bound: 0.0702274
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698265
time: 1.24 seconds

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

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0697266, upper bound: 0.0702543
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700966, upper bound: 0.0698457
time: 2.49 seconds

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

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0698457, upper bound: 0.0700966
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702543, upper bound: 0.0697266
time: 1.61 seconds

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

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0698265, upper bound: 0.0701166
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702274, upper bound: 0.0697385
time: 1.61 seconds

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

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0697385, upper bound: 0.0702274
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698265
time: 1.11 seconds

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

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0697266, upper bound: 0.0702543
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700966, upper bound: 0.0698457
time: 1.58 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0698457, upper bound: 0.0700966
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0702543, upper bound: 0.0697266
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0698265, upper bound: 0.0701166
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0702274, upper bound: 0.0697385
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0697385, upper bound: 0.0702274
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698265
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0697266, upper bound: 0.0702543
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0700966, upper bound: 0.0698457
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0698457, upper bound: 0.0700966
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0702543, upper bound: 0.0697266
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0698265, upper bound: 0.0701166
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0702274, upper bound: 0.0697385
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0697385, upper bound: 0.0702274
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698265
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0697266, upper bound: 0.0702543
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.35
Output dim: 9, lower bound: -0.0700966, upper bound: 0.0698457

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0390442
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665548, upper bound: 0.0668668
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665536, upper bound: 0.0668668
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0388931, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669852, upper bound: 0.0664708
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669852, upper bound: 0.0664763
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0390452
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665145, upper bound: 0.0669122
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665134, upper bound: 0.0669122
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0388837, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669263, upper bound: 0.0664964
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669263, upper bound: 0.0665019
time: 1.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0387885
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665019, upper bound: 0.0669263
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0664964, upper bound: 0.0669263
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0390318
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669122, upper bound: 0.0665134
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669122, upper bound: 0.0665145
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0387939
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0664763, upper bound: 0.0669852
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0664708, upper bound: 0.0669852
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0390276
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0668668, upper bound: 0.0665536
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0668668, upper bound: 0.0665548
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0390276, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665548, upper bound: 0.0668668
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665536, upper bound: 0.0668668
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0387940, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669852, upper bound: 0.0664708
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669852, upper bound: 0.0664763
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0390318, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665145, upper bound: 0.0669122
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665134, upper bound: 0.0669122
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0387885, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669263, upper bound: 0.0664964
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669263, upper bound: 0.0665019
time: 1.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0390224
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0665019, upper bound: 0.0669263
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0664964, upper bound: 0.0669263
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0390452, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669122, upper bound: 0.0665134
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669122, upper bound: 0.0665145
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0390279
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0664763, upper bound: 0.0669852
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0664708, upper bound: 0.0669852
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099
1: -0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0390442, 0.0391061
2: -0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713
3: -0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883
4: -0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926
5: -0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117
6: -0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370
7: -0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255
8: -0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314
9: 0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0668668, upper bound: 0.0665536
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0668668, upper bound: 0.0665548
time: 1.24 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0665548, upper bound: 0.0668668
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0665536, upper bound: 0.0668668
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669852, upper bound: 0.0664708
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669852, upper bound: 0.0664763
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0665145, upper bound: 0.0669122
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0665134, upper bound: 0.0669122
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669263, upper bound: 0.0664964
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669263, upper bound: 0.0665019
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0665019, upper bound: 0.0669263
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0664964, upper bound: 0.0669263
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669122, upper bound: 0.0665134
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669122, upper bound: 0.0665145
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0664763, upper bound: 0.0669852
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0664708, upper bound: 0.0669852
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0668668, upper bound: 0.0665536
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0668668, upper bound: 0.0665548
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0665548, upper bound: 0.0668668
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0665536, upper bound: 0.0668668
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669852, upper bound: 0.0664708
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669852, upper bound: 0.0664763
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0665145, upper bound: 0.0669122
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0665134, upper bound: 0.0669122
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669263, upper bound: 0.0664964
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669263, upper bound: 0.0665019
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0665019, upper bound: 0.0669263
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0664964, upper bound: 0.0669263
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669122, upper bound: 0.0665134
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0669122, upper bound: 0.0665145
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0664763, upper bound: 0.0669852
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0664708, upper bound: 0.0669852
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0668668, upper bound: 0.0665536
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 9, lower bound: -0.0668668, upper bound: 0.0665548

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 9.80 + 148.95 = 158.75 seconds
