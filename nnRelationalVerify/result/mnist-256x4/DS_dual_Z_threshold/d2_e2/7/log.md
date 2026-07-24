## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0134062


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010673, 0.0010673)
1: (-0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0059097, 0.0059097)
2: (0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0132030, 0.0132030)
3: (-0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0055638, 0.0055638)
4: (0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0215853, 0.0215853)
5: (0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041992, 0.0041992)
6: (-0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0054646, 0.0054646)
7: (-0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006971, 0.0006971)
8: (-0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037756, 0.0037756)
9: (-0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0189017, 0.0189017)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 2.44 = 4.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0157720, upper bound: 0.0157720

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0155210, upper bound: 0.0155210
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0155210, upper bound: 0.0155210
time: 1.50 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.08 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.08
Output dim: 4, lower bound: -0.0155210, upper bound: 0.0155210
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.08
Output dim: 4, lower bound: -0.0155210, upper bound: 0.0155210

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010566, 0.0010566
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0058504, 0.0058504
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0130705, 0.0130705
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0055079, 0.0055079
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0213687, 0.0213687
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041571, 0.0041571
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0054098, 0.0054098
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006901, 0.0006901
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037377, 0.0037377
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0187121, 0.0187120

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120432, upper bound: 0.0120432
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120432, upper bound: 0.0120432
time: 0.88 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010566, 0.0010673
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0059097, 0.0058504
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0130705, 0.0132030
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0055638, 0.0055079
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0215853, 0.0213687
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041992, 0.0041571
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0054098, 0.0054646
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006901, 0.0006971
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037756, 0.0037377
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0187120, 0.0189017

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120432, upper bound: 0.0120432
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120432, upper bound: 0.0120432
time: 0.88 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.52 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.52
Output dim: 4, lower bound: -0.0120432, upper bound: 0.0120432
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.52
Output dim: 4, lower bound: -0.0120432, upper bound: 0.0120432
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 3.52
Output dim: 4, lower bound: -0.0120432, upper bound: 0.0120432
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 3.52
Output dim: 4, lower bound: -0.0120432, upper bound: 0.0120432

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.02 + 10.25 = 14.27 seconds
