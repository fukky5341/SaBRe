## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 1.6947652919999998
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548)
1: (-0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906)
2: (-0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625)
3: (-1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869)
4: (-1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024)
5: (-0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151)
6: (-0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975)
7: (-0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587)
8: (-1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157)
9: (-1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295)

## BASE Result
execution time: IAR + LP analysis = 1.48 + 2.76 = 4.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422


# Binary Search by BASE starts (time budget: 1995.76 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary Search Result
Binary search time: 13.66 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1982.11 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.26 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.65
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.65
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.22 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.99
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.85
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.85
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.85
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.85
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.85
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.85
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.85
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.85
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.17 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.89
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.11 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.20 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.15 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.02
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.02
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.02
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.02
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.02
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.02
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.02
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.02
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.14 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.13
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.19 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.59
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.59
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.20 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.21 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.14 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.98 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.98
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.98
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.98
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.98
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.98
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.98
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.98
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.98
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.07 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.75
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.03 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 0.99 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
time: 0.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 5, lower bound: -1.7887082, upper bound: 1.7887082

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
time: 0.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 5, lower bound: -1.7127596, upper bound: 1.7127596

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2409946, 1.0483602, -1.2409946, 1.0483602, -2.2893548, 2.2893548
1: -0.8382131, 0.8810775, -0.8382131, 0.8810775, -1.7192906, 1.7192906
2: -0.9507371, 1.1434255, -0.9507371, 1.1434255, -2.0941625, 2.0941625
3: -1.1043036, 0.7903833, -1.1043036, 0.7903833, -1.8946869, 1.8946869
4: -1.1383610, 1.0591414, -1.1383610, 1.0591414, -2.1975024, 2.1975024
5: -0.7855370, 1.2882781, -0.7855370, 1.2882781, -2.0738151, 2.0738151
6: -0.9008614, 1.0440361, -0.9008614, 1.0440361, -1.9448975, 1.9448975
7: -0.9593127, 1.0878459, -0.9593127, 1.0878459, -2.0471587, 2.0471587
8: -1.2368577, 1.0598582, -1.2368577, 1.0598582, -2.2967157, 2.2967157
9: -1.0177740, 1.0362554, -1.0177740, 1.0362554, -2.0540295, 2.0540295

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
time: 0.99 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.78
Output dim: 5, lower bound: -1.6933911, upper bound: 1.6933911
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 342.06 seconds
