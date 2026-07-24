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
execution time: IAR + LP analysis = 1.29 + 11.11 = 12.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -206.2537789, upper bound: 206.2537789


# Binary Search by BASE starts (time budget: 2687.60 seconds, max iter: 100)

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
Binary search time: 42.32 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2645.27 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147950, upper bound: 206.2147950
time: 7.95 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147950, upper bound: 206.2147950
time: 7.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.81
Output dim: 1, lower bound: -206.2147950, upper bound: 206.2147950
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.81
Output dim: 1, lower bound: -206.2147950, upper bound: 206.2147950

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
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1524008, upper bound: 206.1524008
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1524008, upper bound: 206.1524008
time: 6.81 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147950, upper bound: 206.2147413
time: 7.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147413, upper bound: 206.2147950
time: 6.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.31 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 17.31
Output dim: 1, lower bound: -206.1524008, upper bound: 206.1524008
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 17.31
Output dim: 1, lower bound: -206.1524008, upper bound: 206.1524008
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.31
Output dim: 1, lower bound: -206.2147950, upper bound: 206.2147413
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.31
Output dim: 1, lower bound: -206.2147413, upper bound: 206.2147950

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
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147950, upper bound: 206.2147413
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147950, upper bound: 206.2147413
time: 7.56 seconds

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147382, upper bound: 206.2147949
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147413, upper bound: 206.2147876
time: 7.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.23 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.23
Output dim: 1, lower bound: -206.2147950, upper bound: 206.2147413
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.23
Output dim: 1, lower bound: -206.2147950, upper bound: 206.2147413
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.23
Output dim: 1, lower bound: -206.2147382, upper bound: 206.2147949
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.23
Output dim: 1, lower bound: -206.2147413, upper bound: 206.2147876

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 162

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1552635, upper bound: 206.1552438
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1552635, upper bound: 206.1552438
time: 6.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1576284, upper bound: 206.1576134
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1576284, upper bound: 206.1576134
time: 6.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 234

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147217, upper bound: 206.2147798
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147217, upper bound: 206.2147798
time: 7.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2143770, upper bound: 206.2144207
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2143774, upper bound: 206.2144209
time: 7.96 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.13 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.13
Output dim: 1, lower bound: -206.1552635, upper bound: 206.1552438
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.13
Output dim: 1, lower bound: -206.1552635, upper bound: 206.1552438
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.13
Output dim: 1, lower bound: -206.1576284, upper bound: 206.1576134
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.13
Output dim: 1, lower bound: -206.1576284, upper bound: 206.1576134
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.13
Output dim: 1, lower bound: -206.2147217, upper bound: 206.2147798
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.13
Output dim: 1, lower bound: -206.2147217, upper bound: 206.2147798
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.13
Output dim: 1, lower bound: -206.2143770, upper bound: 206.2144207
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.13
Output dim: 1, lower bound: -206.2143774, upper bound: 206.2144209

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147120, upper bound: 206.2147658
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147041, upper bound: 206.2147650
time: 8.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1640504, upper bound: 206.1640407
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1640504, upper bound: 206.1640407
time: 6.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2143770, upper bound: 206.2143958
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2143468, upper bound: 206.2144207
time: 6.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1802656, upper bound: 206.1802574
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1802656, upper bound: 206.1802574
time: 7.10 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 15.38 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.38
Output dim: 1, lower bound: -206.2147120, upper bound: 206.2147658
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.38
Output dim: 1, lower bound: -206.2147041, upper bound: 206.2147650
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.38
Output dim: 1, lower bound: -206.1640504, upper bound: 206.1640407
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.38
Output dim: 1, lower bound: -206.1640504, upper bound: 206.1640407
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.38
Output dim: 1, lower bound: -206.2143770, upper bound: 206.2143958
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.38
Output dim: 1, lower bound: -206.2143468, upper bound: 206.2144207
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.38
Output dim: 1, lower bound: -206.1802656, upper bound: 206.1802574
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.38
Output dim: 1, lower bound: -206.1802656, upper bound: 206.1802574

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2133237, upper bound: 206.2133427
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2133237, upper bound: 206.2133427
time: 8.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147041, upper bound: 206.2147650
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147040, upper bound: 206.2147567
time: 7.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2079287, upper bound: 206.2079210
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2079323, upper bound: 206.2079169
time: 8.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1902555, upper bound: 206.1902835
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1902555, upper bound: 206.1902835
time: 7.22 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 18.20 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.20
Output dim: 1, lower bound: -206.2133237, upper bound: 206.2133427
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.20
Output dim: 1, lower bound: -206.2133237, upper bound: 206.2133427
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.20
Output dim: 1, lower bound: -206.2147041, upper bound: 206.2147650
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.20
Output dim: 1, lower bound: -206.2147040, upper bound: 206.2147567
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.20
Output dim: 1, lower bound: -206.2079287, upper bound: 206.2079210
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.20
Output dim: 1, lower bound: -206.2079323, upper bound: 206.2079169
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 18.20
Output dim: 1, lower bound: -206.1902555, upper bound: 206.1902835
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 18.20
Output dim: 1, lower bound: -206.1902555, upper bound: 206.1902835

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1898336, upper bound: 206.1898370
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1898336, upper bound: 206.1898370
time: 8.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1631285, upper bound: 206.1631422
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1631285, upper bound: 206.1631422
time: 8.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147041, upper bound: 206.2147650
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2147018, upper bound: 206.2147639
time: 8.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2143666, upper bound: 206.2144096
time: 8.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2143671, upper bound: 206.2144172
time: 9.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2079137, upper bound: 206.2079000
time: 8.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2079137, upper bound: 206.2079000
time: 7.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2065474, upper bound: 206.2065582
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2065474, upper bound: 206.2065582
time: 7.81 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 16.10 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.1898336, upper bound: 206.1898370
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.1898336, upper bound: 206.1898370
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.1631285, upper bound: 206.1631422
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.1631285, upper bound: 206.1631422
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.2147041, upper bound: 206.2147650
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.2147018, upper bound: 206.2147639
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.2143666, upper bound: 206.2144096
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.2143671, upper bound: 206.2144172
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.2079137, upper bound: 206.2079000
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.2079137, upper bound: 206.2079000
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.2065474, upper bound: 206.2065582
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.10
Output dim: 1, lower bound: -206.2065474, upper bound: 206.2065582

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 151

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2143421, upper bound: 206.2143731
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2143421, upper bound: 206.2143731
time: 7.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1744181, upper bound: 206.1743890
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1744181, upper bound: 206.1743890
time: 7.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2129437, upper bound: 206.2129598
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2129437, upper bound: 206.2129598
time: 7.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1649867, upper bound: 206.1649936
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1649867, upper bound: 206.1649936
time: 6.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2007198, upper bound: 206.2007153
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2007198, upper bound: 206.2007153
time: 6.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 151

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2079134, upper bound: 206.2079000
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2079085, upper bound: 206.2079003
time: 6.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1456181, upper bound: 206.1456531
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1456181, upper bound: 206.1456531
time: 6.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2065211, upper bound: 206.2065293
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2065211, upper bound: 206.2065289
time: 7.15 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 15.26 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.2143421, upper bound: 206.2143731
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.2143421, upper bound: 206.2143731
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.1744181, upper bound: 206.1743890
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.1744181, upper bound: 206.1743890
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.2129437, upper bound: 206.2129598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.2129437, upper bound: 206.2129598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.1649867, upper bound: 206.1649936
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.1649867, upper bound: 206.1649936
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.2007198, upper bound: 206.2007153
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.2007198, upper bound: 206.2007153
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.2079134, upper bound: 206.2079000
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.2079085, upper bound: 206.2079003
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.1456181, upper bound: 206.1456531
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.1456181, upper bound: 206.1456531
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.2065211, upper bound: 206.2065293
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 15.26
Output dim: 1, lower bound: -206.2065211, upper bound: 206.2065289

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2128231, upper bound: 206.2128263
time: 7.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2128231, upper bound: 206.2128263
time: 9.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2124227, upper bound: 206.2124312
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2124227, upper bound: 206.2124312
time: 6.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2097551, upper bound: 206.2097756
time: 8.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2097541, upper bound: 206.2097872
time: 8.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1690231, upper bound: 206.1690319
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1690231, upper bound: 206.1690319
time: 6.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2006940, upper bound: 206.2006855
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2006921, upper bound: 206.2006857
time: 9.13 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 16.82 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 16.82
Output dim: 1, lower bound: -206.2128231, upper bound: 206.2128263
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 16.82
Output dim: 1, lower bound: -206.2128231, upper bound: 206.2128263
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 16.82
Output dim: 1, lower bound: -206.2124227, upper bound: 206.2124312
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 16.82
Output dim: 1, lower bound: -206.2124227, upper bound: 206.2124312
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 16.82
Output dim: 1, lower bound: -206.2097551, upper bound: 206.2097756
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 16.82
Output dim: 1, lower bound: -206.2097541, upper bound: 206.2097872
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 16.82
Output dim: 1, lower bound: -206.1690231, upper bound: 206.1690319
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 16.82
Output dim: 1, lower bound: -206.1690231, upper bound: 206.1690319
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 16.82
Output dim: 1, lower bound: -206.2006940, upper bound: 206.2006855
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 16.82
Output dim: 1, lower bound: -206.2006921, upper bound: 206.2006857
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 16.82
Output dim: 1, lower bound: -206.2007198, upper bound: 206.2007153
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 16.82
Output dim: 1, lower bound: -206.2079134, upper bound: 206.2079000
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 16.82
Output dim: 1, lower bound: -206.2079085, upper bound: 206.2079003
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 16.82
Output dim: 1, lower bound: -206.2065211, upper bound: 206.2065293
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 16.82
Output dim: 1, lower bound: -206.2065211, upper bound: 206.2065289
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.24490356445312
rel_dist={1: [-206.25362701135504, 206.25362701135498]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1713209, upper bound: 206.1713209
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1713209, upper bound: 206.1713209
time: 6.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.62 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 13.62
Output dim: 1, lower bound: -206.1713209, upper bound: 206.1713209
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 13.62
Output dim: 1, lower bound: -206.1713209, upper bound: 206.1713209
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=207.24490356445312
rel_dist={1: [-206.2534079160462, 206.25340791604617]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2476403, upper bound: 206.2476403
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2476403, upper bound: 206.2476403
time: 8.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.13
Output dim: 1, lower bound: -206.2476403, upper bound: 206.2476403
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.13
Output dim: 1, lower bound: -206.2476403, upper bound: 206.2476403

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2476403, upper bound: 206.2476331
time: 8.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2476331, upper bound: 206.2476403
time: 9.06 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1915415, upper bound: 206.1915417
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1915415, upper bound: 206.1915417
time: 7.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.73
Output dim: 1, lower bound: -206.2476403, upper bound: 206.2476331
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.73
Output dim: 1, lower bound: -206.2476331, upper bound: 206.2476403
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 15.73
Output dim: 1, lower bound: -206.1915415, upper bound: 206.1915417
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 15.73
Output dim: 1, lower bound: -206.1915415, upper bound: 206.1915417

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2476403, upper bound: 206.2476228
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2476344, upper bound: 206.2476331
time: 10.30 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1913233, upper bound: 206.1913379
time: 7.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1913233, upper bound: 206.1913379
time: 7.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.82 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.82
Output dim: 1, lower bound: -206.2476403, upper bound: 206.2476228
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.82
Output dim: 1, lower bound: -206.2476344, upper bound: 206.2476331
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 16.82
Output dim: 1, lower bound: -206.1913233, upper bound: 206.1913379
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 16.82
Output dim: 1, lower bound: -206.1913233, upper bound: 206.1913379

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2471292, upper bound: 206.2471120
time: 8.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2471283, upper bound: 206.2471119
time: 8.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1814346, upper bound: 206.1814312
time: 8.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1814346, upper bound: 206.1814312
time: 8.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.73
Output dim: 1, lower bound: -206.2471292, upper bound: 206.2471120
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.73
Output dim: 1, lower bound: -206.2471283, upper bound: 206.2471119
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.73
Output dim: 1, lower bound: -206.1814346, upper bound: 206.1814312
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.73
Output dim: 1, lower bound: -206.1814346, upper bound: 206.1814312

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1926447, upper bound: 206.1926570
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1926447, upper bound: 206.1926570
time: 7.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 151

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2452992, upper bound: 206.2452912
time: 8.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2452992, upper bound: 206.2452912
time: 9.86 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 19.99
Output dim: 1, lower bound: -206.1926447, upper bound: 206.1926570
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 19.99
Output dim: 1, lower bound: -206.1926447, upper bound: 206.1926570
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.99
Output dim: 1, lower bound: -206.2452992, upper bound: 206.2452912
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.99
Output dim: 1, lower bound: -206.2452992, upper bound: 206.2452912

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2169977, upper bound: 206.2169961
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2169977, upper bound: 206.2169961
time: 8.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2452952, upper bound: 206.2452912
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2452992, upper bound: 206.2452878
time: 10.07 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 20.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.43
Output dim: 1, lower bound: -206.2169977, upper bound: 206.2169961
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.43
Output dim: 1, lower bound: -206.2169977, upper bound: 206.2169961
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.43
Output dim: 1, lower bound: -206.2452952, upper bound: 206.2452912
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.43
Output dim: 1, lower bound: -206.2452992, upper bound: 206.2452878

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2144868, upper bound: 206.2144910
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2144868, upper bound: 206.2144910
time: 7.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2114212, upper bound: 206.2114235
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2114216, upper bound: 206.2114197
time: 9.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 121

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2452891, upper bound: 206.2452912
time: 9.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2452952, upper bound: 206.2452895
time: 9.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 162

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2133574, upper bound: 206.2133561
time: 7.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2133574, upper bound: 206.2133561
time: 7.73 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 16.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.68
Output dim: 1, lower bound: -206.2144868, upper bound: 206.2144910
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.68
Output dim: 1, lower bound: -206.2144868, upper bound: 206.2144910
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.68
Output dim: 1, lower bound: -206.2114212, upper bound: 206.2114235
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.68
Output dim: 1, lower bound: -206.2114216, upper bound: 206.2114197
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.68
Output dim: 1, lower bound: -206.2452891, upper bound: 206.2452912
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.68
Output dim: 1, lower bound: -206.2452952, upper bound: 206.2452895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.68
Output dim: 1, lower bound: -206.2133574, upper bound: 206.2133561
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.68
Output dim: 1, lower bound: -206.2133574, upper bound: 206.2133561

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2144741, upper bound: 206.2144910
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2144868, upper bound: 206.2144735
time: 9.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1898843, upper bound: 206.1898689
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1898843, upper bound: 206.1898689
time: 8.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2059944, upper bound: 206.2059850
time: 8.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2059944, upper bound: 206.2059850
time: 8.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2114216, upper bound: 206.2114191
time: 10.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2114196, upper bound: 206.2114197
time: 9.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2228649, upper bound: 206.2228562
time: 9.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2228649, upper bound: 206.2228562
time: 9.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2452725, upper bound: 206.2452895
time: 8.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2452952, upper bound: 206.2452743
time: 9.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2109782, upper bound: 206.2109766
time: 9.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2109797, upper bound: 206.2109765
time: 8.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1320780, upper bound: 206.1320519
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1320780, upper bound: 206.1320519
time: 6.90 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 14.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2144741, upper bound: 206.2144910
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2144868, upper bound: 206.2144735
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.1898843, upper bound: 206.1898689
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.1898843, upper bound: 206.1898689
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2059944, upper bound: 206.2059850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2059944, upper bound: 206.2059850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2114216, upper bound: 206.2114191
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2114196, upper bound: 206.2114197
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2228649, upper bound: 206.2228562
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2228649, upper bound: 206.2228562
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2452725, upper bound: 206.2452895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2452952, upper bound: 206.2452743
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2109782, upper bound: 206.2109766
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.2109797, upper bound: 206.2109765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.1320780, upper bound: 206.1320519
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 14.98
Output dim: 1, lower bound: -206.1320780, upper bound: 206.1320519

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1838326, upper bound: 206.1838329
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1838326, upper bound: 206.1838329
time: 8.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1898839, upper bound: 206.1898593
time: 7.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1898839, upper bound: 206.1898596
time: 8.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2037985, upper bound: 206.2037928
time: 9.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2037943, upper bound: 206.2037935
time: 9.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2059847, upper bound: 206.2059850
time: 8.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2059944, upper bound: 206.2059792
time: 8.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2086189, upper bound: 206.2085914
time: 9.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2086189, upper bound: 206.2085914
time: 11.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2062496, upper bound: 206.2062594
time: 8.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2062496, upper bound: 206.2062594
time: 8.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2228649, upper bound: 206.2228539
time: 8.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2228648, upper bound: 206.2228562
time: 8.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1453722, upper bound: 206.1453624
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1453722, upper bound: 206.1453624
time: 6.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1420703, upper bound: 206.1420722
time: 8.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1420703, upper bound: 206.1420722
time: 8.19 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 17.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.1838326, upper bound: 206.1838329
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.1838326, upper bound: 206.1838329
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.1898839, upper bound: 206.1898593
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.1898839, upper bound: 206.1898596
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.2037985, upper bound: 206.2037928
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.2037943, upper bound: 206.2037935
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.2059847, upper bound: 206.2059850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.2059944, upper bound: 206.2059792
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.2086189, upper bound: 206.2085914
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.2086189, upper bound: 206.2085914
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.2062496, upper bound: 206.2062594
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.2062496, upper bound: 206.2062594
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.2228649, upper bound: 206.2228539
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.2228648, upper bound: 206.2228562
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.1453722, upper bound: 206.1453624
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.1453722, upper bound: 206.1453624
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.1420703, upper bound: 206.1420722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 17.63
Output dim: 1, lower bound: -206.1420703, upper bound: 206.1420722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 17.63
Output dim: 1, lower bound: -206.2452952, upper bound: 206.2452743
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 17.63
Output dim: 1, lower bound: -206.2109782, upper bound: 206.2109766
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 17.63
Output dim: 1, lower bound: -206.2109797, upper bound: 206.2109765
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=207.24490356445312
rel_dist={1: [-206.2535233801417, 206.25352337710052]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1243.33 seconds
