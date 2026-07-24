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
execution time: IAR + LP analysis = 1.51 + 2.71 = 4.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422


# Binary Search by BASE starts (time budget: 1995.78 seconds, max iter: 100)

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
Binary search time: 13.82 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1981.96 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6478942, upper bound: 1.6478942
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6478942, upper bound: 1.6478942
time: 1.06 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.14 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 2.14
Output dim: 5, lower bound: -1.6478942, upper bound: 1.6478942
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 2.14
Output dim: 5, lower bound: -1.6478942, upper bound: 1.6478942
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
time: 1.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.58
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.58
Output dim: 5, lower bound: -1.8029422, upper bound: 1.8029422

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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
time: 1.09 seconds

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7957629, upper bound: 1.7957629
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7957629, upper bound: 1.7957629
time: 1.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 5, lower bound: -1.7957629, upper bound: 1.7957629
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 5, lower bound: -1.7957629, upper bound: 1.7957629

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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
time: 1.09 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6285616, upper bound: 1.6285616
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6285616, upper bound: 1.6285616
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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7957629, upper bound: 1.7957629
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7957629, upper bound: 1.7957629
time: 1.36 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
time: 1.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.00
Output dim: 5, lower bound: -1.6285616, upper bound: 1.6285616
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.00
Output dim: 5, lower bound: -1.6285616, upper bound: 1.6285616
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 5, lower bound: -1.7957629, upper bound: 1.7957629
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 5, lower bound: -1.7957629, upper bound: 1.7957629
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6025117, upper bound: 1.6025117
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6025117, upper bound: 1.6025117
time: 0.94 seconds

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7926605, upper bound: 1.7926605
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7926605, upper bound: 1.7926605
time: 1.29 seconds

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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7945891, upper bound: 1.7945891
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7945891, upper bound: 1.7945891
time: 1.18 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
time: 1.31 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
time: 1.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.6025117, upper bound: 1.6025117
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.6025117, upper bound: 1.6025117
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.7926605, upper bound: 1.7926605
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.7926605, upper bound: 1.7926605
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.7945891, upper bound: 1.7945891
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.7945891, upper bound: 1.7945891
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7482493, upper bound: 1.7482493
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7482493, upper bound: 1.7482493
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6664577, upper bound: 1.6664577
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6664577, upper bound: 1.6664577
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7266481, upper bound: 1.7266481
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7266481, upper bound: 1.7266481
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7935252, upper bound: 1.7935252
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7935252, upper bound: 1.7935252
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6588543, upper bound: 1.6588543
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6588543, upper bound: 1.6588543
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6750229, upper bound: 1.6750229
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6750229, upper bound: 1.6750229
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6057744, upper bound: 1.6057744
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6057744, upper bound: 1.6057744
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
time: 1.25 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7482493, upper bound: 1.7482493
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7482493, upper bound: 1.7482493
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7484741, upper bound: 1.7484741
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.6664577, upper bound: 1.6664577
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.6664577, upper bound: 1.6664577
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7266481, upper bound: 1.7266481
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7266481, upper bound: 1.7266481
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7935252, upper bound: 1.7935252
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7935252, upper bound: 1.7935252
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.6588543, upper bound: 1.6588543
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.6588543, upper bound: 1.6588543
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.6750229, upper bound: 1.6750229
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.6750229, upper bound: 1.6750229
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.6057744, upper bound: 1.6057744
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.6057744, upper bound: 1.6057744
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.01
Output dim: 5, lower bound: -1.7931320, upper bound: 1.7931320

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6918069, upper bound: 1.6918069
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6918069, upper bound: 1.6918069
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7468142, upper bound: 1.7468142
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7468142, upper bound: 1.7468142
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6669886, upper bound: 1.6669886
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6669886, upper bound: 1.6669886
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7256484, upper bound: 1.7256484
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7256484, upper bound: 1.7256484
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7003074, upper bound: 1.7003074
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7003074, upper bound: 1.7003074
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6235196, upper bound: 1.6235196
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6235196, upper bound: 1.6235196
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6382451, upper bound: 1.6382451
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6382451, upper bound: 1.6382451
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7592280, upper bound: 1.7592280
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7592280, upper bound: 1.7592280
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6164668, upper bound: 1.6164668
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6164668, upper bound: 1.6164668
time: 1.09 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.6918069, upper bound: 1.6918069
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.6918069, upper bound: 1.6918069
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7468142, upper bound: 1.7468142
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7468142, upper bound: 1.7468142
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.6669886, upper bound: 1.6669886
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.6669886, upper bound: 1.6669886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7256484, upper bound: 1.7256484
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7256484, upper bound: 1.7256484
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7003074, upper bound: 1.7003074
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7003074, upper bound: 1.7003074
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.6235196, upper bound: 1.6235196
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.6235196, upper bound: 1.6235196
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.6382451, upper bound: 1.6382451
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.6382451, upper bound: 1.6382451
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7230134, upper bound: 1.7230134
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7592280, upper bound: 1.7592280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.7592280, upper bound: 1.7592280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.6164668, upper bound: 1.6164668
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 5, lower bound: -1.6164668, upper bound: 1.6164668

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7455505, upper bound: 1.7455505
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7455505, upper bound: 1.7455505
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7447937, upper bound: 1.7447937
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7447937, upper bound: 1.7447937
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7455568, upper bound: 1.7455568
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7455568, upper bound: 1.7455568
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7253522, upper bound: 1.7253522
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7253522, upper bound: 1.7253522
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7227886, upper bound: 1.7227886
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7227886, upper bound: 1.7227886
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5650720, upper bound: 1.5650720
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5650720, upper bound: 1.5650720
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5845115, upper bound: 1.5845115
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5845115, upper bound: 1.5845115
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7117657, upper bound: 1.7117657
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7117657, upper bound: 1.7117657
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7229949, upper bound: 1.7229949
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7229949, upper bound: 1.7229949
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7229949, upper bound: 1.7229949
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7229949, upper bound: 1.7229949
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5880843, upper bound: 1.5880843
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5880843, upper bound: 1.5880843
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7592280, upper bound: 1.7592280
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7592280, upper bound: 1.7592280
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7526915, upper bound: 1.7526915
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7526915, upper bound: 1.7526915
time: 1.24 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7455505, upper bound: 1.7455505
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7455505, upper bound: 1.7455505
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7447937, upper bound: 1.7447937
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7447937, upper bound: 1.7447937
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7455568, upper bound: 1.7455568
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7455568, upper bound: 1.7455568
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7253522, upper bound: 1.7253522
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7253522, upper bound: 1.7253522
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7227886, upper bound: 1.7227886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7227886, upper bound: 1.7227886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.5650720, upper bound: 1.5650720
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.5650720, upper bound: 1.5650720
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.5845115, upper bound: 1.5845115
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.5845115, upper bound: 1.5845115
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7117657, upper bound: 1.7117657
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7117657, upper bound: 1.7117657
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7229949, upper bound: 1.7229949
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7229949, upper bound: 1.7229949
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7229949, upper bound: 1.7229949
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7229949, upper bound: 1.7229949
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.5880843, upper bound: 1.5880843
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.5880843, upper bound: 1.5880843
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7592280, upper bound: 1.7592280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7592280, upper bound: 1.7592280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7526915, upper bound: 1.7526915
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 5, lower bound: -1.7526915, upper bound: 1.7526915

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6024555, upper bound: 1.6024555
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6024555, upper bound: 1.6024555
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6657187, upper bound: 1.6657187
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6657187, upper bound: 1.6657187
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7444166, upper bound: 1.7444166
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7444166, upper bound: 1.7444166
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6261620, upper bound: 1.6261620
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6261620, upper bound: 1.6261620
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6285616, upper bound: 1.6285616
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6285616, upper bound: 1.6285616
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7381027, upper bound: 1.7381027
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7381027, upper bound: 1.7381027
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7169437, upper bound: 1.7169437
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7169437, upper bound: 1.7169437
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7253522, upper bound: 1.7253522
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7253522, upper bound: 1.7253522
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7137863, upper bound: 1.7137863
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7137863, upper bound: 1.7137863
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6090825, upper bound: 1.6090825
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6090825, upper bound: 1.6090825
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7094792, upper bound: 1.7094792
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7094792, upper bound: 1.7094792
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7042267, upper bound: 1.7042267
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7042267, upper bound: 1.7042267
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7225132, upper bound: 1.7225132
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7225132, upper bound: 1.7225132
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7209186, upper bound: 1.7209186
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7209186, upper bound: 1.7209186
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
time: 1.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7166337, upper bound: 1.7166337
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7166337, upper bound: 1.7166337
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7571028, upper bound: 1.7571028
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7571028, upper bound: 1.7571028
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7564048, upper bound: 1.7564048
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7564048, upper bound: 1.7564048
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5729824, upper bound: 1.5729824
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5729824, upper bound: 1.5729824
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7526915, upper bound: 1.7526915
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7526915, upper bound: 1.7526915
time: 1.21 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7464890, upper bound: 1.7464890
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.6024555, upper bound: 1.6024555
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.6024555, upper bound: 1.6024555
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.6657187, upper bound: 1.6657187
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.6657187, upper bound: 1.6657187
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7444166, upper bound: 1.7444166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7444166, upper bound: 1.7444166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.6261620, upper bound: 1.6261620
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.6261620, upper bound: 1.6261620
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.6285616, upper bound: 1.6285616
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.6285616, upper bound: 1.6285616
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7381027, upper bound: 1.7381027
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7381027, upper bound: 1.7381027
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7169437, upper bound: 1.7169437
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7169437, upper bound: 1.7169437
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7253522, upper bound: 1.7253522
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7253522, upper bound: 1.7253522
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7137863, upper bound: 1.7137863
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7137863, upper bound: 1.7137863
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.6090825, upper bound: 1.6090825
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.6090825, upper bound: 1.6090825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7094792, upper bound: 1.7094792
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7094792, upper bound: 1.7094792
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7042267, upper bound: 1.7042267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7042267, upper bound: 1.7042267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7225132, upper bound: 1.7225132
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7225132, upper bound: 1.7225132
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7209186, upper bound: 1.7209186
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7209186, upper bound: 1.7209186
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7166337, upper bound: 1.7166337
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7166337, upper bound: 1.7166337
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7571028, upper bound: 1.7571028
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7571028, upper bound: 1.7571028
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7564048, upper bound: 1.7564048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7564048, upper bound: 1.7564048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.5729824, upper bound: 1.5729824
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.5729824, upper bound: 1.5729824
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7526915, upper bound: 1.7526915
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.87
Output dim: 5, lower bound: -1.7526915, upper bound: 1.7526915

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7325083, upper bound: 1.7325083
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7325083, upper bound: 1.7325083
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7436248, upper bound: 1.7436248
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7436248, upper bound: 1.7436248
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7444166, upper bound: 1.7444166
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7444166, upper bound: 1.7444166
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7368146, upper bound: 1.7368146
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7368146, upper bound: 1.7368146
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7381027, upper bound: 1.7381027
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7381027, upper bound: 1.7381027
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7120953, upper bound: 1.7120953
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7120953, upper bound: 1.7120953
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6021310, upper bound: 1.6021310
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6021310, upper bound: 1.6021310
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7134090, upper bound: 1.7134090
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7134090, upper bound: 1.7134090
time: 1.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7220220, upper bound: 1.7220220
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7220220, upper bound: 1.7220220
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7216323, upper bound: 1.7216323
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7216323, upper bound: 1.7216323
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7136961, upper bound: 1.7136961
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7136961, upper bound: 1.7136961
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7132673, upper bound: 1.7132673
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7132673, upper bound: 1.7132673
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5756047, upper bound: 1.5756047
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5756047, upper bound: 1.5756047
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7029211, upper bound: 1.7029211
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7029211, upper bound: 1.7029211
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7042267, upper bound: 1.7042267
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7042267, upper bound: 1.7042267
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6612646, upper bound: 1.6612646
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6612646, upper bound: 1.6612646
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7225132, upper bound: 1.7225132
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7225132, upper bound: 1.7225132
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7211580, upper bound: 1.7211580
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7211580, upper bound: 1.7211580
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7194015, upper bound: 1.7194015
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7194015, upper bound: 1.7194015
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5856959, upper bound: 1.5856959
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5856959, upper bound: 1.5856959
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7166337, upper bound: 1.7166337
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7166337, upper bound: 1.7166337
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6719675, upper bound: 1.6719675
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6719675, upper bound: 1.6719675
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7523813, upper bound: 1.7523813
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7523813, upper bound: 1.7523813
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7566718, upper bound: 1.7566718
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7566718, upper bound: 1.7566718
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6708702, upper bound: 1.6708702
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6708702, upper bound: 1.6708702
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7545958, upper bound: 1.7545958
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7545958, upper bound: 1.7545958
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7525409, upper bound: 1.7525409
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7525409, upper bound: 1.7525409
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7477418, upper bound: 1.7477418
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7477418, upper bound: 1.7477418
time: 1.24 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 8.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7325083, upper bound: 1.7325083
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7325083, upper bound: 1.7325083
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7436248, upper bound: 1.7436248
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7436248, upper bound: 1.7436248
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7444166, upper bound: 1.7444166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7444166, upper bound: 1.7444166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7368146, upper bound: 1.7368146
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7368146, upper bound: 1.7368146
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7381027, upper bound: 1.7381027
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7381027, upper bound: 1.7381027
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7120953, upper bound: 1.7120953
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7120953, upper bound: 1.7120953
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.6021310, upper bound: 1.6021310
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.6021310, upper bound: 1.6021310
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7134090, upper bound: 1.7134090
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7134090, upper bound: 1.7134090
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7220220, upper bound: 1.7220220
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7220220, upper bound: 1.7220220
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7216323, upper bound: 1.7216323
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7216323, upper bound: 1.7216323
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7136961, upper bound: 1.7136961
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7136961, upper bound: 1.7136961
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7132673, upper bound: 1.7132673
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7132673, upper bound: 1.7132673
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.5756047, upper bound: 1.5756047
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.5756047, upper bound: 1.5756047
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7029211, upper bound: 1.7029211
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7029211, upper bound: 1.7029211
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7042267, upper bound: 1.7042267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7042267, upper bound: 1.7042267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.6612646, upper bound: 1.6612646
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.6612646, upper bound: 1.6612646
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7225132, upper bound: 1.7225132
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7225132, upper bound: 1.7225132
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7211580, upper bound: 1.7211580
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7211580, upper bound: 1.7211580
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7194015, upper bound: 1.7194015
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7194015, upper bound: 1.7194015
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.5856959, upper bound: 1.5856959
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.5856959, upper bound: 1.5856959
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7216406, upper bound: 1.7216406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7166337, upper bound: 1.7166337
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7166337, upper bound: 1.7166337
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.6719675, upper bound: 1.6719675
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.6719675, upper bound: 1.6719675
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7523813, upper bound: 1.7523813
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7523813, upper bound: 1.7523813
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7566718, upper bound: 1.7566718
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7566718, upper bound: 1.7566718
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.6708702, upper bound: 1.6708702
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.6708702, upper bound: 1.6708702
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7545958, upper bound: 1.7545958
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7545958, upper bound: 1.7545958
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7525409, upper bound: 1.7525409
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7525409, upper bound: 1.7525409
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7477418, upper bound: 1.7477418
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 8.31
Output dim: 5, lower bound: -1.7477418, upper bound: 1.7477418

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7385392, upper bound: 1.7385392
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7385392, upper bound: 1.7385392
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6313308, upper bound: 1.6313308
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6313308, upper bound: 1.6313308
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6175357, upper bound: 1.6175357
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6175357, upper bound: 1.6175357
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6802653, upper bound: 1.6802653
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6802653, upper bound: 1.6802653
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7389948, upper bound: 1.7389948
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6643169, upper bound: 1.6643169
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6643169, upper bound: 1.6643169
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6256484, upper bound: 1.6256484
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6256484, upper bound: 1.6256484
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7079911, upper bound: 1.7079911
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7079911, upper bound: 1.7079911
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7444166, upper bound: 1.7444166
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7444166, upper bound: 1.7444166
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7368146, upper bound: 1.7368146
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7368146, upper bound: 1.7368146
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary search (step 2) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6657805, upper bound: 1.6657805
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6657805, upper bound: 1.6657805
time: 1.07 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.16 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 2.16
Output dim: 5, lower bound: -1.6657805, upper bound: 1.6657805
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 2.16
Output dim: 5, lower bound: -1.6657805, upper bound: 1.6657805
Binary search (step 2): status=Status.VERIFIED, k_low=7, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary search (step 3) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7935513, upper bound: 1.7935513
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7935513, upper bound: 1.7935513
time: 1.13 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.31 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.31
Output dim: 5, lower bound: -1.7935513, upper bound: 1.7935513
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.31
Output dim: 5, lower bound: -1.7935513, upper bound: 1.7935513

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7231891, upper bound: 1.7231891
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7231891, upper bound: 1.7231891
time: 1.09 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371040, upper bound: 1.7371040
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371040, upper bound: 1.7371040
time: 1.09 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 5, lower bound: -1.7231891, upper bound: 1.7231891
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 5, lower bound: -1.7231891, upper bound: 1.7231891
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 5, lower bound: -1.7371040, upper bound: 1.7371040
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 5, lower bound: -1.7371040, upper bound: 1.7371040

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7153772, upper bound: 1.7153772
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7153772, upper bound: 1.7153772
time: 1.05 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7194155, upper bound: 1.7194155
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7194155, upper bound: 1.7194155
time: 1.04 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371040, upper bound: 1.7371040
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371040, upper bound: 1.7371040
time: 1.11 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.17 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.66 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.66
Output dim: 5, lower bound: -1.7153772, upper bound: 1.7153772
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.66
Output dim: 5, lower bound: -1.7153772, upper bound: 1.7153772
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.66
Output dim: 5, lower bound: -1.7194155, upper bound: 1.7194155
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.66
Output dim: 5, lower bound: -1.7194155, upper bound: 1.7194155
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.66
Output dim: 5, lower bound: -1.7371040, upper bound: 1.7371040
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.66
Output dim: 5, lower bound: -1.7371040, upper bound: 1.7371040
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.66
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.66
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6523158, upper bound: 1.6523158
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6523158, upper bound: 1.6523158
time: 1.06 seconds

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.12 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.09 seconds

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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7104007, upper bound: 1.7104007
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7104007, upper bound: 1.7104007
time: 1.06 seconds

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7325223, upper bound: 1.7325223
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7325223, upper bound: 1.7325223
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.06 seconds

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6959468, upper bound: 1.6959468
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6959468, upper bound: 1.6959468
time: 1.10 seconds

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
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.21 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.6523158, upper bound: 1.6523158
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.6523158, upper bound: 1.6523158
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7104007, upper bound: 1.7104007
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7104007, upper bound: 1.7104007
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7325223, upper bound: 1.7325223
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7325223, upper bound: 1.7325223
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.6959468, upper bound: 1.6959468
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.6959468, upper bound: 1.6959468
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.84
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7044954, upper bound: 1.7044954
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7044954, upper bound: 1.7044954
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7039912, upper bound: 1.7039912
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7039912, upper bound: 1.7039912
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6776568, upper bound: 1.6776568
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6776568, upper bound: 1.6776568
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7288125, upper bound: 1.7288125
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7288125, upper bound: 1.7288125
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6959468, upper bound: 1.6959468
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6959468, upper bound: 1.6959468
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6861615, upper bound: 1.6861615
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6861615, upper bound: 1.6861615
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.20 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.76 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7044954, upper bound: 1.7044954
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7044954, upper bound: 1.7044954
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7039912, upper bound: 1.7039912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7039912, upper bound: 1.7039912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.6776568, upper bound: 1.6776568
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.6776568, upper bound: 1.6776568
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7288125, upper bound: 1.7288125
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7288125, upper bound: 1.7288125
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.6959468, upper bound: 1.6959468
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.6959468, upper bound: 1.6959468
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.6861615, upper bound: 1.6861615
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.6861615, upper bound: 1.6861615
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6996780, upper bound: 1.6996780
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6996780, upper bound: 1.6996780
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7054918, upper bound: 1.7054918
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7054918, upper bound: 1.7054918
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7001986, upper bound: 1.7001986
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7001986, upper bound: 1.7001986
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6390917, upper bound: 1.6390917
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6390917, upper bound: 1.6390917
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6420434, upper bound: 1.6420434
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6420434, upper bound: 1.6420434
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5448928, upper bound: 1.5448928
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5448928, upper bound: 1.5448928
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7039912, upper bound: 1.7039912
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7039912, upper bound: 1.7039912
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7036379, upper bound: 1.7036379
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7036379, upper bound: 1.7036379
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7311125, upper bound: 1.7311125
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7311125, upper bound: 1.7311125
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5860530, upper bound: 1.5860530
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5860530, upper bound: 1.5860530
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7278806, upper bound: 1.7278806
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7278806, upper bound: 1.7278806
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6898856, upper bound: 1.6898856
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6898856, upper bound: 1.6898856
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6914794, upper bound: 1.6914794
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6914794, upper bound: 1.6914794
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7316863, upper bound: 1.7316863
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7316863, upper bound: 1.7316863
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7040393, upper bound: 1.7040393
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7040393, upper bound: 1.7040393
time: 1.17 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.79 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.6996780, upper bound: 1.6996780
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.6996780, upper bound: 1.6996780
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7054918, upper bound: 1.7054918
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7054918, upper bound: 1.7054918
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7001986, upper bound: 1.7001986
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7001986, upper bound: 1.7001986
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.6390917, upper bound: 1.6390917
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.6390917, upper bound: 1.6390917
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.6420434, upper bound: 1.6420434
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.6420434, upper bound: 1.6420434
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.5448928, upper bound: 1.5448928
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.5448928, upper bound: 1.5448928
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7039912, upper bound: 1.7039912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7039912, upper bound: 1.7039912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7036379, upper bound: 1.7036379
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7036379, upper bound: 1.7036379
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7311125, upper bound: 1.7311125
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7311125, upper bound: 1.7311125
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.5860530, upper bound: 1.5860530
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.5860530, upper bound: 1.5860530
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7278806, upper bound: 1.7278806
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7278806, upper bound: 1.7278806
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.6898856, upper bound: 1.6898856
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.6898856, upper bound: 1.6898856
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.6914794, upper bound: 1.6914794
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.6914794, upper bound: 1.6914794
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7316863, upper bound: 1.7316863
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7316863, upper bound: 1.7316863
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7040393, upper bound: 1.7040393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.79
Output dim: 5, lower bound: -1.7040393, upper bound: 1.7040393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6996780, upper bound: 1.6996780
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6996780, upper bound: 1.6996780
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6957521, upper bound: 1.6957521
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6957521, upper bound: 1.6957521
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6995691, upper bound: 1.6995691
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6995691, upper bound: 1.6995691
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6422787, upper bound: 1.6422787
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6422787, upper bound: 1.6422787
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5310261, upper bound: 1.5310261
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5310261, upper bound: 1.5310261
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6939099, upper bound: 1.6939099
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6939099, upper bound: 1.6939099
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7001986, upper bound: 1.7001986
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7001986, upper bound: 1.7001986
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7057015, upper bound: 1.7057015
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7057015, upper bound: 1.7057015
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6743043, upper bound: 1.6743043
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6743043, upper bound: 1.6743043
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7053357, upper bound: 1.7053357
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7053357, upper bound: 1.7053357
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6947410, upper bound: 1.6947410
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6947410, upper bound: 1.6947410
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7035341, upper bound: 1.7035341
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7035341, upper bound: 1.7035341
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6289208, upper bound: 1.6289208
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6289208, upper bound: 1.6289208
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6943790, upper bound: 1.6943790
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6943790, upper bound: 1.6943790
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5792324, upper bound: 1.5792324
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5792324, upper bound: 1.5792324
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5791971, upper bound: 1.5791971
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.5791971, upper bound: 1.5791971
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7292223, upper bound: 1.7292223
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7292223, upper bound: 1.7292223
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7278806, upper bound: 1.7278806
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7278806, upper bound: 1.7278806
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6850211, upper bound: 1.6850211
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6850211, upper bound: 1.6850211
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7283024, upper bound: 1.7283024
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7283024, upper bound: 1.7283024
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6914794, upper bound: 1.6914794
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6914794, upper bound: 1.6914794
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7318294, upper bound: 1.7318294
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7318294, upper bound: 1.7318294
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6763159, upper bound: 1.6763159
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6763159, upper bound: 1.6763159
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7283024, upper bound: 1.7283024
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7283024, upper bound: 1.7283024
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6293524, upper bound: 1.6293524
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6293524, upper bound: 1.6293524
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7040393, upper bound: 1.7040393
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7040393, upper bound: 1.7040393
time: 1.17 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 6.05 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6996780, upper bound: 1.6996780
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6996780, upper bound: 1.6996780
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6957521, upper bound: 1.6957521
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6957521, upper bound: 1.6957521
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6995691, upper bound: 1.6995691
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6995691, upper bound: 1.6995691
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6422787, upper bound: 1.6422787
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6422787, upper bound: 1.6422787
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.5310261, upper bound: 1.5310261
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.5310261, upper bound: 1.5310261
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6939099, upper bound: 1.6939099
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6939099, upper bound: 1.6939099
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7001986, upper bound: 1.7001986
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7001986, upper bound: 1.7001986
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7057015, upper bound: 1.7057015
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7057015, upper bound: 1.7057015
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6743043, upper bound: 1.6743043
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6743043, upper bound: 1.6743043
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7053357, upper bound: 1.7053357
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7053357, upper bound: 1.7053357
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6947410, upper bound: 1.6947410
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6947410, upper bound: 1.6947410
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7035341, upper bound: 1.7035341
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7035341, upper bound: 1.7035341
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6289208, upper bound: 1.6289208
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6289208, upper bound: 1.6289208
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6943790, upper bound: 1.6943790
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6943790, upper bound: 1.6943790
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.5792324, upper bound: 1.5792324
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.5792324, upper bound: 1.5792324
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7315841, upper bound: 1.7315841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.5791971, upper bound: 1.5791971
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.5791971, upper bound: 1.5791971
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7292223, upper bound: 1.7292223
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7292223, upper bound: 1.7292223
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7371038, upper bound: 1.7371038
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7212277, upper bound: 1.7212277
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7278806, upper bound: 1.7278806
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7278806, upper bound: 1.7278806
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6850211, upper bound: 1.6850211
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6850211, upper bound: 1.6850211
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7337215, upper bound: 1.7337215
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7283024, upper bound: 1.7283024
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7283024, upper bound: 1.7283024
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6914794, upper bound: 1.6914794
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6914794, upper bound: 1.6914794
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7318294, upper bound: 1.7318294
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7318294, upper bound: 1.7318294
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6763159, upper bound: 1.6763159
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6763159, upper bound: 1.6763159
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7283024, upper bound: 1.7283024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7283024, upper bound: 1.7283024
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6293524, upper bound: 1.6293524
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.6293524, upper bound: 1.6293524
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7040393, upper bound: 1.7040393
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.05
Output dim: 5, lower bound: -1.7040393, upper bound: 1.7040393

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6568106, upper bound: 1.6568106
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6568106, upper bound: 1.6568106
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6986166, upper bound: 1.6986166
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6986166, upper bound: 1.6986166
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6313508, upper bound: 1.6313508
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6313508, upper bound: 1.6313508
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6527715, upper bound: 1.6527715
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6527715, upper bound: 1.6527715
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6549108, upper bound: 1.6549108
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6549108, upper bound: 1.6549108
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6985095, upper bound: 1.6985095
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.6985095, upper bound: 1.6985095
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6564839, upper bound: 1.6564839
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6564839, upper bound: 1.6564839
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6564839, upper bound: 1.6564839
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6564839, upper bound: 1.6564839
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7056608, upper bound: 1.7056608
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6382102, upper bound: 1.6382102
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6382102, upper bound: 1.6382102
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6390917, upper bound: 1.6390917
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.6390917, upper bound: 1.6390917
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7131567, upper bound: 1.7131567
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7131567, upper bound: 1.7131567
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7073335, upper bound: 1.7073335
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7073335, upper bound: 1.7073335
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7141118, upper bound: 1.7141118
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7137053, upper bound: 1.7137053
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7057015, upper bound: 1.7057015
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.7057015, upper bound: 1.7057015
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 3): status=Status.UNKNOWN, k_low=8, k_high=8, k_mid=8, eps_mid=0.0312500, abs_max=2.073815107345581
rel_dist={5: [-1.8029421580625726, 1.8029421580625726]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.02734375
execution time: 1215.75 seconds
