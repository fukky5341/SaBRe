## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.783554394


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.1714453, 1.0967407, 0.1714453, 1.0967407, -0.9252955, 0.9252955)
1: (-0.4131228, 0.4088053, -0.4131228, 0.4088053, -0.8219281, 0.8219281)
2: (-0.3177431, 0.5175457, -0.3177431, 0.5175457, -0.8352888, 0.8352888)
3: (-0.2996281, 0.3978117, -0.2996281, 0.3978117, -0.6974397, 0.6974397)
4: (-0.4314196, 0.4241707, -0.4314196, 0.4241707, -0.8555903, 0.8555903)
5: (-0.4635416, 0.5828649, -0.4635416, 0.5828649, -1.0464065, 1.0464065)
6: (-0.3379162, 0.4767012, -0.3379162, 0.4767012, -0.8146174, 0.8146174)
7: (-0.4291549, 0.4561338, -0.4291549, 0.4561338, -0.8852887, 0.8852887)
8: (-0.4519421, 0.5508475, -0.4519421, 0.5508475, -1.0027897, 1.0027897)
9: (-0.4255759, 0.5267283, -0.4255759, 0.5267283, -0.9523041, 0.9523041)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.87 + 2.78 = 3.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7995453

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7995453
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7995453
time: 1.63 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.19 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.19
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7995453
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.19
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7995453

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.1714453, 1.0967407, 0.1714453, 1.0967407, -0.9252955, 0.9252955
1: -0.4131228, 0.4088053, -0.4131228, 0.4088053, -0.8219281, 0.8219281
2: -0.3177431, 0.5175457, -0.3177431, 0.5175457, -0.8352888, 0.8352888
3: -0.2996281, 0.3978117, -0.2996281, 0.3978117, -0.6974397, 0.6974397
4: -0.4314196, 0.4241707, -0.4314196, 0.4241707, -0.8555903, 0.8555903
5: -0.4635416, 0.5828649, -0.4635416, 0.5828649, -1.0464065, 1.0464065
6: -0.3379162, 0.4767012, -0.3379162, 0.4767012, -0.8146174, 0.8146174
7: -0.4291549, 0.4561338, -0.4291549, 0.4561338, -0.8852887, 0.8852887
8: -0.4519421, 0.5508475, -0.4519421, 0.5508475, -1.0027897, 1.0027897
9: -0.4255759, 0.5267283, -0.4255759, 0.5267283, -0.9523041, 0.9523041

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7961189, upper bound: 0.7961189
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7961189, upper bound: 0.7961189
time: 1.74 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.1714453, 1.0967407, 0.1714453, 1.0967407, -0.9252955, 0.9252955
1: -0.4131228, 0.4088053, -0.4131228, 0.4088053, -0.8219281, 0.8219281
2: -0.3177431, 0.5175457, -0.3177431, 0.5175457, -0.8352888, 0.8352888
3: -0.2996281, 0.3978117, -0.2996281, 0.3978117, -0.6974397, 0.6974397
4: -0.4314196, 0.4241707, -0.4314196, 0.4241707, -0.8555903, 0.8555903
5: -0.4635416, 0.5828649, -0.4635416, 0.5828649, -1.0464065, 1.0464065
6: -0.3379162, 0.4767012, -0.3379162, 0.4767012, -0.8146174, 0.8146174
7: -0.4291549, 0.4561338, -0.4291549, 0.4561338, -0.8852887, 0.8852887
8: -0.4519421, 0.5508475, -0.4519421, 0.5508475, -1.0027897, 1.0027897
9: -0.4255759, 0.5267283, -0.4255759, 0.5267283, -0.9523041, 0.9523041

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7842355, upper bound: 0.7842355
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7842355, upper bound: 0.7842355
time: 1.19 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.27 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -0.7961189, upper bound: 0.7961189
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -0.7961189, upper bound: 0.7961189
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -0.7842355, upper bound: 0.7842355
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -0.7842355, upper bound: 0.7842355

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.1714453, 1.0967407, 0.1714453, 1.0967407, -0.9252955, 0.9252955
1: -0.4131228, 0.4088053, -0.4131228, 0.4088053, -0.8219281, 0.8219281
2: -0.3177431, 0.5175457, -0.3177431, 0.5175457, -0.8352888, 0.8352888
3: -0.2996281, 0.3978117, -0.2996281, 0.3978117, -0.6974397, 0.6974397
4: -0.4314196, 0.4241707, -0.4314196, 0.4241707, -0.8555903, 0.8555903
5: -0.4635416, 0.5828649, -0.4635416, 0.5828649, -1.0464065, 1.0464065
6: -0.3379162, 0.4767012, -0.3379162, 0.4767012, -0.8146174, 0.8146174
7: -0.4291549, 0.4561338, -0.4291549, 0.4561338, -0.8852887, 0.8852887
8: -0.4519421, 0.5508475, -0.4519421, 0.5508475, -1.0027897, 1.0027897
9: -0.4255759, 0.5267283, -0.4255759, 0.5267283, -0.9523041, 0.9523041

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7827092, upper bound: 0.7827092
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7827092, upper bound: 0.7827092
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.1714453, 1.0967407, 0.1714453, 1.0967407, -0.9252955, 0.9252955
1: -0.4131228, 0.4088053, -0.4131228, 0.4088053, -0.8219281, 0.8219281
2: -0.3177431, 0.5175457, -0.3177431, 0.5175457, -0.8352888, 0.8352888
3: -0.2996281, 0.3978117, -0.2996281, 0.3978117, -0.6974397, 0.6974397
4: -0.4314196, 0.4241707, -0.4314196, 0.4241707, -0.8555903, 0.8555903
5: -0.4635416, 0.5828649, -0.4635416, 0.5828649, -1.0464065, 1.0464065
6: -0.3379162, 0.4767012, -0.3379162, 0.4767012, -0.8146174, 0.8146174
7: -0.4291549, 0.4561338, -0.4291549, 0.4561338, -0.8852887, 0.8852887
8: -0.4519421, 0.5508475, -0.4519421, 0.5508475, -1.0027897, 1.0027897
9: -0.4255759, 0.5267283, -0.4255759, 0.5267283, -0.9523041, 0.9523041

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7802795, upper bound: 0.7802795
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7802795, upper bound: 0.7802795
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.1714453, 1.0967407, 0.1714453, 1.0967407, -0.9252955, 0.9252955
1: -0.4131228, 0.4088053, -0.4131228, 0.4088053, -0.8219281, 0.8219281
2: -0.3177431, 0.5175457, -0.3177431, 0.5175457, -0.8352888, 0.8352888
3: -0.2996281, 0.3978117, -0.2996281, 0.3978117, -0.6974397, 0.6974397
4: -0.4314196, 0.4241707, -0.4314196, 0.4241707, -0.8555903, 0.8555903
5: -0.4635416, 0.5828649, -0.4635416, 0.5828649, -1.0464065, 1.0464065
6: -0.3379162, 0.4767012, -0.3379162, 0.4767012, -0.8146174, 0.8146174
7: -0.4291549, 0.4561338, -0.4291549, 0.4561338, -0.8852887, 0.8852887
8: -0.4519421, 0.5508475, -0.4519421, 0.5508475, -1.0027897, 1.0027897
9: -0.4255759, 0.5267283, -0.4255759, 0.5267283, -0.9523041, 0.9523041

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7696629, upper bound: 0.7696629
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7696629, upper bound: 0.7696629
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.1714453, 1.0967407, 0.1714453, 1.0967407, -0.9252955, 0.9252955
1: -0.4131228, 0.4088053, -0.4131228, 0.4088053, -0.8219281, 0.8219281
2: -0.3177431, 0.5175457, -0.3177431, 0.5175457, -0.8352888, 0.8352888
3: -0.2996281, 0.3978117, -0.2996281, 0.3978117, -0.6974397, 0.6974397
4: -0.4314196, 0.4241707, -0.4314196, 0.4241707, -0.8555903, 0.8555903
5: -0.4635416, 0.5828649, -0.4635416, 0.5828649, -1.0464065, 1.0464065
6: -0.3379162, 0.4767012, -0.3379162, 0.4767012, -0.8146174, 0.8146174
7: -0.4291549, 0.4561338, -0.4291549, 0.4561338, -0.8852887, 0.8852887
8: -0.4519421, 0.5508475, -0.4519421, 0.5508475, -1.0027897, 1.0027897
9: -0.4255759, 0.5267283, -0.4255759, 0.5267283, -0.9523041, 0.9523041

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 88

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7842355, upper bound: 0.7842355
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7842355, upper bound: 0.7842355
time: 1.14 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.13 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.7827092, upper bound: 0.7827092
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.7827092, upper bound: 0.7827092
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.7802795, upper bound: 0.7802795
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.7802795, upper bound: 0.7802795
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.7696629, upper bound: 0.7696629
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.7696629, upper bound: 0.7696629
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.7842355, upper bound: 0.7842355
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.7842355, upper bound: 0.7842355

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.1714453, 1.0967407, 0.1714453, 1.0967407, -0.9252955, 0.9252955
1: -0.4131228, 0.4088053, -0.4131228, 0.4088053, -0.8219281, 0.8219281
2: -0.3177431, 0.5175457, -0.3177431, 0.5175457, -0.8352888, 0.8352888
3: -0.2996281, 0.3978117, -0.2996281, 0.3978117, -0.6974397, 0.6974397
4: -0.4314196, 0.4241707, -0.4314196, 0.4241707, -0.8555903, 0.8555903
5: -0.4635416, 0.5828649, -0.4635416, 0.5828649, -1.0464065, 1.0464065
6: -0.3379162, 0.4767012, -0.3379162, 0.4767012, -0.8146174, 0.8146174
7: -0.4291549, 0.4561338, -0.4291549, 0.4561338, -0.8852887, 0.8852887
8: -0.4519421, 0.5508475, -0.4519421, 0.5508475, -1.0027897, 1.0027897
9: -0.4255759, 0.5267283, -0.4255759, 0.5267283, -0.9523041, 0.9523041

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7720780, upper bound: 0.7720780
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7720780, upper bound: 0.7720780
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.1714453, 1.0967407, 0.1714453, 1.0967407, -0.9252955, 0.9252955
1: -0.4131228, 0.4088053, -0.4131228, 0.4088053, -0.8219281, 0.8219281
2: -0.3177431, 0.5175457, -0.3177431, 0.5175457, -0.8352888, 0.8352888
3: -0.2996281, 0.3978117, -0.2996281, 0.3978117, -0.6974397, 0.6974397
4: -0.4314196, 0.4241707, -0.4314196, 0.4241707, -0.8555903, 0.8555903
5: -0.4635416, 0.5828649, -0.4635416, 0.5828649, -1.0464065, 1.0464065
6: -0.3379162, 0.4767012, -0.3379162, 0.4767012, -0.8146174, 0.8146174
7: -0.4291549, 0.4561338, -0.4291549, 0.4561338, -0.8852887, 0.8852887
8: -0.4519421, 0.5508475, -0.4519421, 0.5508475, -1.0027897, 1.0027897
9: -0.4255759, 0.5267283, -0.4255759, 0.5267283, -0.9523041, 0.9523041

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7758065, upper bound: 0.7758065
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7758065, upper bound: 0.7758065
time: 1.10 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.00 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.7720780, upper bound: 0.7720780
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.7720780, upper bound: 0.7720780
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.7758065, upper bound: 0.7758065
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.7758065, upper bound: 0.7758065

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.65 + 30.34 = 33.99 seconds
