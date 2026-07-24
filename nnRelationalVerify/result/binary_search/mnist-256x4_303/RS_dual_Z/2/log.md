## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 206.199692099
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469)
1: (-113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036)
2: (-147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656)
3: (-156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682)
4: (-143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783)
5: (-127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586)
6: (-122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480)
7: (-134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603)
8: (-162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730)
9: (-122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292)

## BASE Result
execution time: IAR + LP analysis = 1.29 + 11.07 = 12.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -206.2537789, upper bound: 206.2537789


# Binary Search by BASE starts (time budget: 2687.64 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.24490356445312
rel_dist={1: [-206.25362701135504, 206.25362701135498]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=207.24490356445312
rel_dist={1: [-206.2534079160462, 206.25340791604617]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=207.24490356445312
rel_dist={1: [-206.25298865227455, 206.2529886525199]}

## Binary Search Result
Binary search time: 42.10 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2645.55 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2009963, upper bound: 206.2009961
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2009963, upper bound: 206.2009961
time: 6.86 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.83
Output dim: 1, lower bound: -206.2009963, upper bound: 206.2009961
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.83
Output dim: 1, lower bound: -206.2009963, upper bound: 206.2009961

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000194, upper bound: 206.2000094
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000094, upper bound: 206.2000194
time: 8.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000194, upper bound: 206.2000094
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000094, upper bound: 206.2000194
time: 8.13 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 1, lower bound: -206.2000194, upper bound: 206.2000094
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 1, lower bound: -206.2000094, upper bound: 206.2000194
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 1, lower bound: -206.2000194, upper bound: 206.2000094
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 1, lower bound: -206.2000094, upper bound: 206.2000194

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994857, upper bound: 206.1994794
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994857, upper bound: 206.1994794
time: 6.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994792, upper bound: 206.1994859
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994792, upper bound: 206.1994859
time: 5.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994857, upper bound: 206.1994794
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994857, upper bound: 206.1994794
time: 6.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994792, upper bound: 206.1994859
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994792, upper bound: 206.1994859
time: 5.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.25
Output dim: 1, lower bound: -206.1994857, upper bound: 206.1994794
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.25
Output dim: 1, lower bound: -206.1994857, upper bound: 206.1994794
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.25
Output dim: 1, lower bound: -206.1994792, upper bound: 206.1994859
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.25
Output dim: 1, lower bound: -206.1994792, upper bound: 206.1994859
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.25
Output dim: 1, lower bound: -206.1994857, upper bound: 206.1994794
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.25
Output dim: 1, lower bound: -206.1994857, upper bound: 206.1994794
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.25
Output dim: 1, lower bound: -206.1994792, upper bound: 206.1994859
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.25
Output dim: 1, lower bound: -206.1994792, upper bound: 206.1994859
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.24490356445312
rel_dist={1: [-206.25362701135504, 206.25362701135498]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2010123, upper bound: 206.2010120
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2010123, upper bound: 206.2010120
time: 7.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.33 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.33
Output dim: 1, lower bound: -206.2010123, upper bound: 206.2010120
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.33
Output dim: 1, lower bound: -206.2010123, upper bound: 206.2010120

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000419, upper bound: 206.2000283
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000286, upper bound: 206.2000419
time: 6.80 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000419, upper bound: 206.2000283
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000286, upper bound: 206.2000419
time: 6.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.69 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.69
Output dim: 1, lower bound: -206.2000419, upper bound: 206.2000283
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.69
Output dim: 1, lower bound: -206.2000286, upper bound: 206.2000419
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.69
Output dim: 1, lower bound: -206.2000419, upper bound: 206.2000283
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.69
Output dim: 1, lower bound: -206.2000286, upper bound: 206.2000419

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995040, upper bound: 206.1994935
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995040, upper bound: 206.1994935
time: 6.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994936, upper bound: 206.1995040
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994936, upper bound: 206.1995040
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995040, upper bound: 206.1994935
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995040, upper bound: 206.1994935
time: 6.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994936, upper bound: 206.1995040
time: 7.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1994936, upper bound: 206.1995040
time: 6.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 22.16
Output dim: 1, lower bound: -206.1995040, upper bound: 206.1994935
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 22.16
Output dim: 1, lower bound: -206.1995040, upper bound: 206.1994935
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 22.16
Output dim: 1, lower bound: -206.1994936, upper bound: 206.1995040
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 22.16
Output dim: 1, lower bound: -206.1994936, upper bound: 206.1995040
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 22.16
Output dim: 1, lower bound: -206.1995040, upper bound: 206.1994935
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 22.16
Output dim: 1, lower bound: -206.1995040, upper bound: 206.1994935
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 22.16
Output dim: 1, lower bound: -206.1994936, upper bound: 206.1995040
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 22.16
Output dim: 1, lower bound: -206.1994936, upper bound: 206.1995040
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=207.24490356445312
rel_dist={1: [-206.2537045440331, 206.2537045440331]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2010224, upper bound: 206.2010222
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2010224, upper bound: 206.2010222
time: 6.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.69 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.69
Output dim: 1, lower bound: -206.2010224, upper bound: 206.2010222
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.69
Output dim: 1, lower bound: -206.2010224, upper bound: 206.2010222

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000565, upper bound: 206.2000406
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000408, upper bound: 206.2000565
time: 6.21 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000565, upper bound: 206.2000406
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000408, upper bound: 206.2000565
time: 6.21 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.03
Output dim: 1, lower bound: -206.2000565, upper bound: 206.2000406
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.03
Output dim: 1, lower bound: -206.2000408, upper bound: 206.2000565
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.03
Output dim: 1, lower bound: -206.2000565, upper bound: 206.2000406
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.03
Output dim: 1, lower bound: -206.2000408, upper bound: 206.2000565

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995152, upper bound: 206.1995027
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995152, upper bound: 206.1995027
time: 5.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995027, upper bound: 206.1995152
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995027, upper bound: 206.1995152
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995152, upper bound: 206.1995027
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995152, upper bound: 206.1995027
time: 5.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995027, upper bound: 206.1995152
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995027, upper bound: 206.1995152
time: 6.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.36 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.36
Output dim: 1, lower bound: -206.1995152, upper bound: 206.1995027
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.36
Output dim: 1, lower bound: -206.1995152, upper bound: 206.1995027
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.36
Output dim: 1, lower bound: -206.1995027, upper bound: 206.1995152
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.36
Output dim: 1, lower bound: -206.1995027, upper bound: 206.1995152
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.36
Output dim: 1, lower bound: -206.1995152, upper bound: 206.1995027
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.36
Output dim: 1, lower bound: -206.1995152, upper bound: 206.1995027
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.36
Output dim: 1, lower bound: -206.1995027, upper bound: 206.1995152
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.36
Output dim: 1, lower bound: -206.1995027, upper bound: 206.1995152
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=207.24490356445312
rel_dist={1: [-206.25375458972235, 206.25375461649787]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2010269, upper bound: 206.2010269
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2010269, upper bound: 206.2010269
time: 6.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.19 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.19
Output dim: 1, lower bound: -206.2010269, upper bound: 206.2010269
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.19
Output dim: 1, lower bound: -206.2010269, upper bound: 206.2010269

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000632, upper bound: 206.2000468
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000466, upper bound: 206.2000632
time: 6.23 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000632, upper bound: 206.2000468
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2000466, upper bound: 206.2000632
time: 6.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.78
Output dim: 1, lower bound: -206.2000632, upper bound: 206.2000468
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.78
Output dim: 1, lower bound: -206.2000466, upper bound: 206.2000632
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.78
Output dim: 1, lower bound: -206.2000632, upper bound: 206.2000468
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.78
Output dim: 1, lower bound: -206.2000466, upper bound: 206.2000632

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995202, upper bound: 206.1995070
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995202, upper bound: 206.1995071
time: 6.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995070, upper bound: 206.1995202
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995070, upper bound: 206.1995202
time: 6.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995202, upper bound: 206.1995070
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995202, upper bound: 206.1995071
time: 6.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469
1: -113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036
2: -147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656
3: -156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682
4: -143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783
5: -127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586
6: -122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480
7: -134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603
8: -162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730
9: -122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995070, upper bound: 206.1995202
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1995070, upper bound: 206.1995202
time: 6.70 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.83 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.83
Output dim: 1, lower bound: -206.1995202, upper bound: 206.1995070
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.83
Output dim: 1, lower bound: -206.1995202, upper bound: 206.1995071
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.83
Output dim: 1, lower bound: -206.1995070, upper bound: 206.1995202
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.83
Output dim: 1, lower bound: -206.1995070, upper bound: 206.1995202
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.83
Output dim: 1, lower bound: -206.1995202, upper bound: 206.1995070
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.83
Output dim: 1, lower bound: -206.1995202, upper bound: 206.1995071
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 20.83
Output dim: 1, lower bound: -206.1995070, upper bound: 206.1995202
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 20.83
Output dim: 1, lower bound: -206.1995070, upper bound: 206.1995202
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=207.24490356445312
rel_dist={1: [-206.25377892868022, 206.25377892868028]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 609.87 seconds
