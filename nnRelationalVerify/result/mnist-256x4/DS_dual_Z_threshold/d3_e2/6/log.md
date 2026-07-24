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
execution time: IAR + RelationalAnalysis = 2.03 + 2.92 = 4.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7995453

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7962940, upper bound: 0.7962940
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7962940, upper bound: 0.7962940
time: 1.43 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.27 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.27
Output dim: 0, lower bound: -0.7962940, upper bound: 0.7962940
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.27
Output dim: 0, lower bound: -0.7962940, upper bound: 0.7962940

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

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7961986, upper bound: 0.7961986
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7961986, upper bound: 0.7961986
time: 1.39 seconds

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

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7961986, upper bound: 0.7961986
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7961986, upper bound: 0.7961986
time: 1.36 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.99 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 0, lower bound: -0.7961986, upper bound: 0.7961986
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 0, lower bound: -0.7961986, upper bound: 0.7961986
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 0, lower bound: -0.7961986, upper bound: 0.7961986
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.99
Output dim: 0, lower bound: -0.7961986, upper bound: 0.7961986

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

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
time: 1.44 seconds

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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
time: 1.96 seconds

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

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
time: 1.40 seconds

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

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
time: 1.51 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.26 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.26
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.26
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.26
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.26
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.26
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.26
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.26
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.26
Output dim: 0, lower bound: -0.7905570, upper bound: 0.7905570

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848935
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848935
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.54 seconds

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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.69 seconds

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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848935
time: 1.53 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848935
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848935
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848934
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.40
Output dim: 0, lower bound: -0.7848935, upper bound: 0.7848935

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
time: 1.41 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.86 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.86
Output dim: 0, lower bound: -0.7786942, upper bound: 0.7786942

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.95 + 157.18 = 162.13 seconds
