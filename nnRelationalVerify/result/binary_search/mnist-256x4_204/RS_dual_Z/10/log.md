## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 318.144348814
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519)
1: (-174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401)
2: (-227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989)
3: (-242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914)
4: (-222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943)
5: (-198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390)
6: (-190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228)
7: (-207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249)
8: (-250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678)
9: (-188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716)

## BASE Result
execution time: IAR + LP analysis = 1.66 + 11.47 = 13.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -318.2353807, upper bound: 318.2353807


# Binary Search by BASE starts (time budget: 2686.87 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=319.7423400878906
rel_dist={1: [-318.2353263992982, 318.2353263992983]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=319.7423400878906
rel_dist={1: [-318.23529533371016, 318.23529533356407]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=319.7423400878906
rel_dist={1: [-318.2352719030525, 318.23527182273534]}

## Binary Search Result
Binary search time: 44.26 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2642.61 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2235917, upper bound: 318.2235913
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2235917, upper bound: 318.2235913
time: 7.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.10
Output dim: 1, lower bound: -318.2235917, upper bound: 318.2235913
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.10
Output dim: 1, lower bound: -318.2235917, upper bound: 318.2235913

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1466538, upper bound: 318.1466538
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1466538, upper bound: 318.1466538
time: 6.97 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1466538, upper bound: 318.1466538
time: 7.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1466538, upper bound: 318.1466538
time: 7.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.88
Output dim: 1, lower bound: -318.1466538, upper bound: 318.1466538
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.88
Output dim: 1, lower bound: -318.1466538, upper bound: 318.1466538
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.88
Output dim: 1, lower bound: -318.1466538, upper bound: 318.1466538
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.88
Output dim: 1, lower bound: -318.1466538, upper bound: 318.1466538

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462578, upper bound: 318.1462449
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462448, upper bound: 318.1462578
time: 7.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462578, upper bound: 318.1462449
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462448, upper bound: 318.1462578
time: 7.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462578, upper bound: 318.1462449
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462448, upper bound: 318.1462578
time: 7.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462578, upper bound: 318.1462449
time: 7.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462448, upper bound: 318.1462578
time: 7.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.05
Output dim: 1, lower bound: -318.1462578, upper bound: 318.1462449
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.05
Output dim: 1, lower bound: -318.1462448, upper bound: 318.1462578
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.05
Output dim: 1, lower bound: -318.1462578, upper bound: 318.1462449
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.05
Output dim: 1, lower bound: -318.1462448, upper bound: 318.1462578
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.05
Output dim: 1, lower bound: -318.1462578, upper bound: 318.1462449
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.05
Output dim: 1, lower bound: -318.1462448, upper bound: 318.1462578
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.05
Output dim: 1, lower bound: -318.1462578, upper bound: 318.1462449
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.05
Output dim: 1, lower bound: -318.1462448, upper bound: 318.1462578

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
time: 6.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622
time: 7.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
time: 6.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622
time: 7.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
time: 6.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622
time: 7.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
time: 6.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622
time: 7.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.38
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445607, upper bound: 318.1445513
time: 6.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445501
time: 5.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445501, upper bound: 318.1445623
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
time: 7.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445513, upper bound: 318.1445607
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622
time: 6.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445607, upper bound: 318.1445513
time: 6.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445501
time: 5.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445501, upper bound: 318.1445623
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
time: 7.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445513, upper bound: 318.1445607
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622
time: 6.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445607, upper bound: 318.1445513
time: 6.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445501
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445501, upper bound: 318.1445623
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445511, upper bound: 318.1445638
time: 7.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445513, upper bound: 318.1445607
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445520, upper bound: 318.1445622
time: 7.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445520
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445607, upper bound: 318.1445513
time: 6.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445638, upper bound: 318.1445511
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445622, upper bound: 318.1445501
time: 5.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=319.7423400878906
rel_dist={1: [-318.2353263992982, 318.2353263992983]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2234940, upper bound: 318.2234939
time: 9.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2234940, upper bound: 318.2234935
time: 9.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.54
Output dim: 1, lower bound: -318.2234940, upper bound: 318.2234939
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.54
Output dim: 1, lower bound: -318.2234940, upper bound: 318.2234935

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464446
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464446
time: 7.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464446
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464446
time: 7.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.87 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.87
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464446
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.87
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464446
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.87
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464446
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.87
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464446

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1461513, upper bound: 318.1461456
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1461456, upper bound: 318.1461512
time: 6.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1461513, upper bound: 318.1461456
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1461456, upper bound: 318.1461512
time: 6.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1461513, upper bound: 318.1461456
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1461456, upper bound: 318.1461512
time: 6.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1461513, upper bound: 318.1461456
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1461456, upper bound: 318.1461512
time: 6.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.60 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.60
Output dim: 1, lower bound: -318.1461513, upper bound: 318.1461456
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.60
Output dim: 1, lower bound: -318.1461456, upper bound: 318.1461512
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.60
Output dim: 1, lower bound: -318.1461513, upper bound: 318.1461456
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.60
Output dim: 1, lower bound: -318.1461456, upper bound: 318.1461512
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.60
Output dim: 1, lower bound: -318.1461513, upper bound: 318.1461456
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.60
Output dim: 1, lower bound: -318.1461456, upper bound: 318.1461512
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.60
Output dim: 1, lower bound: -318.1461513, upper bound: 318.1461456
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.60
Output dim: 1, lower bound: -318.1461456, upper bound: 318.1461512

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445134
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445133
time: 6.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445196
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445185
time: 6.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445134
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445133
time: 6.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445196
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445185
time: 6.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445134
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445133
time: 6.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445196
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445185
time: 6.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445134
time: 7.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445133
time: 6.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445196
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445185
time: 6.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445134
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445133
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445196
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445185
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445134
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445133
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445196
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445185
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445134
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445133
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445196
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445185
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445134
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445133
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445196
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.99
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445185

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445121
time: 7.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445162, upper bound: 318.1445134
time: 8.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445122
time: 9.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445171, upper bound: 318.1445133
time: 7.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445170
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445122, upper bound: 318.1445196
time: 8.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445162
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445121, upper bound: 318.1445185
time: 10.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445121
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445162, upper bound: 318.1445134
time: 8.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445122
time: 9.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445171, upper bound: 318.1445133
time: 7.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445170
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445122, upper bound: 318.1445196
time: 7.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445162
time: 8.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445121, upper bound: 318.1445185
time: 10.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445121
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445162, upper bound: 318.1445134
time: 8.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445122
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445171, upper bound: 318.1445133
time: 7.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445170
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445122, upper bound: 318.1445196
time: 8.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445162
time: 8.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445121, upper bound: 318.1445185
time: 10.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445121
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1445162, upper bound: 318.1445134
time: 8.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445121
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445162, upper bound: 318.1445134
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445122
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445171, upper bound: 318.1445133
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445170
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445122, upper bound: 318.1445196
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445162
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445121, upper bound: 318.1445185
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445121
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445162, upper bound: 318.1445134
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445122
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445171, upper bound: 318.1445133
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445170
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445122, upper bound: 318.1445196
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445162
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445121, upper bound: 318.1445185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445121
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445162, upper bound: 318.1445134
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445122
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445171, upper bound: 318.1445133
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445170
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445122, upper bound: 318.1445196
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445162
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445121, upper bound: 318.1445185
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445186, upper bound: 318.1445121
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.87
Output dim: 1, lower bound: -318.1445162, upper bound: 318.1445134
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 1, lower bound: -318.1445196, upper bound: 318.1445133
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 1, lower bound: -318.1445133, upper bound: 318.1445196
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.87
Output dim: 1, lower bound: -318.1445134, upper bound: 318.1445185
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=319.7423400878906
rel_dist={1: [-318.23529533371016, 318.23529533356407]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2233793, upper bound: 318.2233788
time: 10.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2233793, upper bound: 318.2233788
time: 10.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.59
Output dim: 1, lower bound: -318.2233793, upper bound: 318.2233788
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.59
Output dim: 1, lower bound: -318.2233793, upper bound: 318.2233788

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460623, upper bound: 318.1460622
time: 8.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460623, upper bound: 318.1460622
time: 8.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460623, upper bound: 318.1460622
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460623, upper bound: 318.1460622
time: 8.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.23
Output dim: 1, lower bound: -318.1460623, upper bound: 318.1460622
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.23
Output dim: 1, lower bound: -318.1460623, upper bound: 318.1460622
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.23
Output dim: 1, lower bound: -318.1460623, upper bound: 318.1460622
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.23
Output dim: 1, lower bound: -318.1460623, upper bound: 318.1460622

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

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
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1459951, upper bound: 318.1459934
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1459934, upper bound: 318.1459950
time: 7.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1459951, upper bound: 318.1459934
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1459934, upper bound: 318.1459950
time: 7.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1459951, upper bound: 318.1459934
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1459934, upper bound: 318.1459950
time: 7.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1459951, upper bound: 318.1459934
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1459934, upper bound: 318.1459950
time: 7.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.32
Output dim: 1, lower bound: -318.1459951, upper bound: 318.1459934
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.32
Output dim: 1, lower bound: -318.1459934, upper bound: 318.1459950
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.32
Output dim: 1, lower bound: -318.1459951, upper bound: 318.1459934
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.32
Output dim: 1, lower bound: -318.1459934, upper bound: 318.1459950
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.32
Output dim: 1, lower bound: -318.1459951, upper bound: 318.1459934
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.32
Output dim: 1, lower bound: -318.1459934, upper bound: 318.1459950
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.32
Output dim: 1, lower bound: -318.1459951, upper bound: 318.1459934
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.32
Output dim: 1, lower bound: -318.1459934, upper bound: 318.1459950

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443170, upper bound: 318.1443161
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443172, upper bound: 318.1443160
time: 7.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443160, upper bound: 318.1443172
time: 10.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443161, upper bound: 318.1443169
time: 9.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443170, upper bound: 318.1443161
time: 8.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443172, upper bound: 318.1443160
time: 7.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443160, upper bound: 318.1443172
time: 9.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443161, upper bound: 318.1443169
time: 9.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443170, upper bound: 318.1443161
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443172, upper bound: 318.1443160
time: 7.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443160, upper bound: 318.1443172
time: 8.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443161, upper bound: 318.1443169
time: 9.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443170, upper bound: 318.1443161
time: 8.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443172, upper bound: 318.1443160
time: 7.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443160, upper bound: 318.1443172
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1443161, upper bound: 318.1443169
time: 9.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443170, upper bound: 318.1443161
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443172, upper bound: 318.1443160
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443160, upper bound: 318.1443172
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443161, upper bound: 318.1443169
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443170, upper bound: 318.1443161
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443172, upper bound: 318.1443160
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443160, upper bound: 318.1443172
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443161, upper bound: 318.1443169
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443170, upper bound: 318.1443161
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443172, upper bound: 318.1443160
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443160, upper bound: 318.1443172
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443161, upper bound: 318.1443169
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443170, upper bound: 318.1443161
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443172, upper bound: 318.1443160
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443160, upper bound: 318.1443172
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.67
Output dim: 1, lower bound: -318.1443161, upper bound: 318.1443169
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=319.7423400878906
rel_dist={1: [-318.2352719030525, 318.23527182273534]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2234468, upper bound: 318.2234464
time: 9.91 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2234468, upper bound: 318.2234464
time: 10.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 20.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 20.50
Output dim: 1, lower bound: -318.2234468, upper bound: 318.2234464
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 20.50
Output dim: 1, lower bound: -318.2234468, upper bound: 318.2234464

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462673, upper bound: 318.1462672
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462673, upper bound: 318.1462673
time: 8.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462673, upper bound: 318.1462673
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462673, upper bound: 318.1462673
time: 8.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.05
Output dim: 1, lower bound: -318.1462673, upper bound: 318.1462672
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.05
Output dim: 1, lower bound: -318.1462673, upper bound: 318.1462673
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.05
Output dim: 1, lower bound: -318.1462673, upper bound: 318.1462673
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.05
Output dim: 1, lower bound: -318.1462673, upper bound: 318.1462673

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

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
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460745, upper bound: 318.1460708
time: 7.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460708, upper bound: 318.1460745
time: 7.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

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
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460745, upper bound: 318.1460708
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460708, upper bound: 318.1460745
time: 7.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460745, upper bound: 318.1460708
time: 7.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460708, upper bound: 318.1460745
time: 7.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460745, upper bound: 318.1460708
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460708, upper bound: 318.1460745
time: 7.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.50 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.50
Output dim: 1, lower bound: -318.1460745, upper bound: 318.1460708
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.50
Output dim: 1, lower bound: -318.1460708, upper bound: 318.1460745
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.50
Output dim: 1, lower bound: -318.1460745, upper bound: 318.1460708
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.50
Output dim: 1, lower bound: -318.1460708, upper bound: 318.1460745
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.50
Output dim: 1, lower bound: -318.1460745, upper bound: 318.1460708
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.50
Output dim: 1, lower bound: -318.1460708, upper bound: 318.1460745
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.50
Output dim: 1, lower bound: -318.1460745, upper bound: 318.1460708
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.50
Output dim: 1, lower bound: -318.1460708, upper bound: 318.1460745

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444323
time: 8.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444323
time: 7.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444363
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444357
time: 9.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444323
time: 8.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444323
time: 7.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444363
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444357
time: 9.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444323
time: 8.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444323
time: 7.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444363
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444357
time: 9.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444323
time: 8.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444323
time: 7.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444363
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444357
time: 9.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444323
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444323
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444363
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444357
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444323
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444323
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444363
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444357
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444323
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444323
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444363
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444357
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444323
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444323
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444363
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444357

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444310
time: 7.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444339, upper bound: 318.1444323
time: 8.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444312
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444344, upper bound: 318.1444322
time: 7.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444343
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444312, upper bound: 318.1444363
time: 9.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444338
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444311, upper bound: 318.1444356
time: 7.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444310
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444339, upper bound: 318.1444323
time: 8.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444312
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444344, upper bound: 318.1444322
time: 7.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444343
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444312, upper bound: 318.1444363
time: 9.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444338
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444311, upper bound: 318.1444356
time: 7.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444310
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444339, upper bound: 318.1444323
time: 8.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444312
time: 7.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444344, upper bound: 318.1444322
time: 7.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444343
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444312, upper bound: 318.1444363
time: 8.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519
1: -174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401
2: -227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989
3: -242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914
4: -222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943
5: -198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390
6: -190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228
7: -207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249
8: -250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678
9: -188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444338
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444311, upper bound: 318.1444356
time: 7.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444310
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444339, upper bound: 318.1444323
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444312
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444344, upper bound: 318.1444322
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444343
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444312, upper bound: 318.1444363
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444338
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444311, upper bound: 318.1444356
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444310
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444339, upper bound: 318.1444323
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444312
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444344, upper bound: 318.1444322
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444343
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444312, upper bound: 318.1444363
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444338
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444311, upper bound: 318.1444356
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444310
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444339, upper bound: 318.1444323
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444312
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444344, upper bound: 318.1444322
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444343
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444312, upper bound: 318.1444363
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444338
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 1, lower bound: -318.1444311, upper bound: 318.1444356
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.65
Output dim: 1, lower bound: -318.1444357, upper bound: 318.1444323
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.65
Output dim: 1, lower bound: -318.1444363, upper bound: 318.1444323
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.65
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444363
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.65
Output dim: 1, lower bound: -318.1444323, upper bound: 318.1444357
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=319.7423400878906
rel_dist={1: [-318.2352845890764, 318.23528450813205]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 2160.00 seconds
