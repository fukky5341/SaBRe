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
execution time: IAR + LP analysis = 1.36 + 11.02 = 12.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -318.2353807, upper bound: 318.2353807


# Binary Search by BASE starts (time budget: 2687.63 seconds, max iter: 100)

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
Binary search time: 44.70 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2642.93 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2353205, upper bound: 318.2353262
time: 8.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2353263, upper bound: 318.2353205
time: 8.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.52
Output dim: 1, lower bound: -318.2353205, upper bound: 318.2353262
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.52
Output dim: 1, lower bound: -318.2353263, upper bound: 318.2353205

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 193

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2327806, upper bound: 318.2327840
time: 9.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2327806, upper bound: 318.2327840
time: 8.51 seconds

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
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 244

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2335574, upper bound: 318.2335471
time: 8.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2335574, upper bound: 318.2335471
time: 9.18 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.39
Output dim: 1, lower bound: -318.2327806, upper bound: 318.2327840
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.39
Output dim: 1, lower bound: -318.2327806, upper bound: 318.2327840
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.39
Output dim: 1, lower bound: -318.2335574, upper bound: 318.2335471
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.39
Output dim: 1, lower bound: -318.2335574, upper bound: 318.2335471

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1983512, upper bound: 318.1983652
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1983512, upper bound: 318.1983652
time: 8.04 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 203

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1902171, upper bound: 318.1902233
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1902171, upper bound: 318.1902233
time: 7.09 seconds

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2282649, upper bound: 318.2282615
time: 9.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2282648, upper bound: 318.2282614
time: 8.93 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2176291, upper bound: 318.2176229
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2176292, upper bound: 318.2176229
time: 8.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 1, lower bound: -318.1983512, upper bound: 318.1983652
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 1, lower bound: -318.1983512, upper bound: 318.1983652
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 1, lower bound: -318.1902171, upper bound: 318.1902233
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 1, lower bound: -318.1902171, upper bound: 318.1902233
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 1, lower bound: -318.2282649, upper bound: 318.2282615
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 1, lower bound: -318.2282648, upper bound: 318.2282614
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 1, lower bound: -318.2176291, upper bound: 318.2176229
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.39
Output dim: 1, lower bound: -318.2176292, upper bound: 318.2176229

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 121

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1937199, upper bound: 318.1937400
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1937356, upper bound: 318.1937199
time: 7.79 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1467930, upper bound: 318.1467982
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1467930, upper bound: 318.1467982
time: 6.76 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1139806, upper bound: 318.1139819
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1139806, upper bound: 318.1139819
time: 6.76 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1824817, upper bound: 318.1824816
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1824817, upper bound: 318.1824816
time: 11.54 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 121

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2282649, upper bound: 318.2282615
time: 10.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2282641, upper bound: 318.2282612
time: 8.06 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 238

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2282649, upper bound: 318.2282615
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2282649, upper bound: 318.2282614
time: 8.10 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2169070, upper bound: 318.2169035
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2169098, upper bound: 318.2169016
time: 8.26 seconds

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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2112136, upper bound: 318.2112131
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2112136, upper bound: 318.2112131
time: 7.26 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.1937199, upper bound: 318.1937400
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.1937356, upper bound: 318.1937199
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.1467930, upper bound: 318.1467982
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.1467930, upper bound: 318.1467982
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.1139806, upper bound: 318.1139819
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.1139806, upper bound: 318.1139819
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.1824817, upper bound: 318.1824816
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.1824817, upper bound: 318.1824816
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.2282649, upper bound: 318.2282615
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.2282641, upper bound: 318.2282612
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.2282649, upper bound: 318.2282615
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.2282649, upper bound: 318.2282614
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.2169070, upper bound: 318.2169035
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.2169098, upper bound: 318.2169016
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.2112136, upper bound: 318.2112131
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.22
Output dim: 1, lower bound: -318.2112136, upper bound: 318.2112131

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1916865, upper bound: 318.1917173
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1916886, upper bound: 318.1917196
time: 7.54 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 151

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1937350, upper bound: 318.1937199
time: 8.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1937356, upper bound: 318.1937198
time: 7.97 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1415151, upper bound: 318.1415317
time: 9.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1415309, upper bound: 318.1415160
time: 7.59 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1107081, upper bound: 318.1107122
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1107081, upper bound: 318.1107122
time: 6.82 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1824767, upper bound: 318.1824815
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1824809, upper bound: 318.1824782
time: 8.11 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462374, upper bound: 318.1462456
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1462374, upper bound: 318.1462456
time: 7.95 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1968815, upper bound: 318.1968593
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1968815, upper bound: 318.1968593
time: 8.21 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1591161, upper bound: 318.1591146
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1591161, upper bound: 318.1591146
time: 7.62 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2259955, upper bound: 318.2259850
time: 9.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2259866, upper bound: 318.2259926
time: 9.10 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2213174, upper bound: 318.2213095
time: 9.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2213174, upper bound: 318.2213095
time: 8.77 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1703071, upper bound: 318.1703031
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1703071, upper bound: 318.1703031
time: 7.30 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1633889, upper bound: 318.1633715
time: 8.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1633889, upper bound: 318.1633715
time: 8.02 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2081184, upper bound: 318.2081159
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2081229, upper bound: 318.2081145
time: 8.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1681170, upper bound: 318.1680926
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1681170, upper bound: 318.1680926
time: 7.52 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1916865, upper bound: 318.1917173
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1916886, upper bound: 318.1917196
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1937350, upper bound: 318.1937199
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1937356, upper bound: 318.1937198
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1415151, upper bound: 318.1415317
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1415309, upper bound: 318.1415160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1107081, upper bound: 318.1107122
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1107081, upper bound: 318.1107122
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1824767, upper bound: 318.1824815
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1824809, upper bound: 318.1824782
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1462374, upper bound: 318.1462456
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1462374, upper bound: 318.1462456
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1968815, upper bound: 318.1968593
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1968815, upper bound: 318.1968593
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1591161, upper bound: 318.1591146
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1591161, upper bound: 318.1591146
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.2259955, upper bound: 318.2259850
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.2259866, upper bound: 318.2259926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.2213174, upper bound: 318.2213095
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.2213174, upper bound: 318.2213095
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1703071, upper bound: 318.1703031
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1703071, upper bound: 318.1703031
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1633889, upper bound: 318.1633715
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1633889, upper bound: 318.1633715
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.2081184, upper bound: 318.2081159
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.2081229, upper bound: 318.2081145
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1681170, upper bound: 318.1680926
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.30
Output dim: 1, lower bound: -318.1681170, upper bound: 318.1680926

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1477037, upper bound: 318.1477108
time: 7.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1477037, upper bound: 318.1477108
time: 7.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1550422, upper bound: 318.1550401
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1550422, upper bound: 318.1550400
time: 7.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1697655, upper bound: 318.1697531
time: 7.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1697655, upper bound: 318.1697531
time: 6.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1648090, upper bound: 318.1648038
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1648090, upper bound: 318.1648038
time: 7.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1822471, upper bound: 318.1822767
time: 8.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1822751, upper bound: 318.1822526
time: 8.47 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 18.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.54
Output dim: 1, lower bound: -318.1477037, upper bound: 318.1477108
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.54
Output dim: 1, lower bound: -318.1477037, upper bound: 318.1477108
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.54
Output dim: 1, lower bound: -318.1550422, upper bound: 318.1550401
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.54
Output dim: 1, lower bound: -318.1550422, upper bound: 318.1550400
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.54
Output dim: 1, lower bound: -318.1697655, upper bound: 318.1697531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.54
Output dim: 1, lower bound: -318.1697655, upper bound: 318.1697531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.54
Output dim: 1, lower bound: -318.1648090, upper bound: 318.1648038
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.54
Output dim: 1, lower bound: -318.1648090, upper bound: 318.1648038
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.54
Output dim: 1, lower bound: -318.1822471, upper bound: 318.1822767
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.54
Output dim: 1, lower bound: -318.1822751, upper bound: 318.1822526
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1824809, upper bound: 318.1824782
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1462374, upper bound: 318.1462456
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1462374, upper bound: 318.1462456
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1968815, upper bound: 318.1968593
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1968815, upper bound: 318.1968593
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1591161, upper bound: 318.1591146
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1591161, upper bound: 318.1591146
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.2259955, upper bound: 318.2259850
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.2259866, upper bound: 318.2259926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.2213174, upper bound: 318.2213095
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.2213174, upper bound: 318.2213095
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1703071, upper bound: 318.1703031
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1703071, upper bound: 318.1703031
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1633889, upper bound: 318.1633715
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1633889, upper bound: 318.1633715
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.2081184, upper bound: 318.2081159
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.2081229, upper bound: 318.2081145
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1681170, upper bound: 318.1680926
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.54
Output dim: 1, lower bound: -318.1681170, upper bound: 318.1680926
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=319.7423400878906
rel_dist={1: [-318.2353263992982, 318.2353263992983]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2296143, upper bound: 318.2296142
time: 9.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2296143, upper bound: 318.2296142
time: 9.01 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.65
Output dim: 1, lower bound: -318.2296143, upper bound: 318.2296142
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.65
Output dim: 1, lower bound: -318.2296143, upper bound: 318.2296142

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 67

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2169683, upper bound: 318.2169683
time: 9.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2169683, upper bound: 318.2169683
time: 9.44 seconds

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
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1211950, upper bound: 318.1211949
time: 7.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1211950, upper bound: 318.1211949
time: 7.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.62 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.62
Output dim: 1, lower bound: -318.2169683, upper bound: 318.2169683
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.62
Output dim: 1, lower bound: -318.2169683, upper bound: 318.2169683
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 16.62
Output dim: 1, lower bound: -318.1211950, upper bound: 318.1211949
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 16.62
Output dim: 1, lower bound: -318.1211950, upper bound: 318.1211949

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1980988, upper bound: 318.1980988
time: 9.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1980988, upper bound: 318.1980988
time: 7.46 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2114093, upper bound: 318.2114093
time: 8.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2114093, upper bound: 318.2114093
time: 8.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.91 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.91
Output dim: 1, lower bound: -318.1980988, upper bound: 318.1980988
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.91
Output dim: 1, lower bound: -318.1980988, upper bound: 318.1980988
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.91
Output dim: 1, lower bound: -318.2114093, upper bound: 318.2114093
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.91
Output dim: 1, lower bound: -318.2114093, upper bound: 318.2114093

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
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1619458, upper bound: 318.1619458
time: 10.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1619458, upper bound: 318.1619458
time: 8.97 seconds

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
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1931154, upper bound: 318.1931154
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1931154, upper bound: 318.1931154
time: 7.09 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2066276, upper bound: 318.2066276
time: 11.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2066276, upper bound: 318.2066276
time: 10.07 seconds

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2066099, upper bound: 318.2066099
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2066099, upper bound: 318.2066099
time: 8.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.03
Output dim: 1, lower bound: -318.1619458, upper bound: 318.1619458
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.03
Output dim: 1, lower bound: -318.1619458, upper bound: 318.1619458
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.03
Output dim: 1, lower bound: -318.1931154, upper bound: 318.1931154
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.03
Output dim: 1, lower bound: -318.1931154, upper bound: 318.1931154
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.03
Output dim: 1, lower bound: -318.2066276, upper bound: 318.2066276
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.03
Output dim: 1, lower bound: -318.2066276, upper bound: 318.2066276
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.03
Output dim: 1, lower bound: -318.2066099, upper bound: 318.2066099
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.03
Output dim: 1, lower bound: -318.2066099, upper bound: 318.2066099

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.0981322, upper bound: 318.0981322
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.0981322, upper bound: 318.0981322
time: 8.08 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1613266, upper bound: 318.1613311
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1613311, upper bound: 318.1613266
time: 8.76 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 151

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1868091, upper bound: 318.1868088
time: 7.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1868091, upper bound: 318.1868088
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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1693169, upper bound: 318.1693169
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1693169, upper bound: 318.1693169
time: 7.34 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1923117, upper bound: 318.1923115
time: 10.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1923117, upper bound: 318.1923115
time: 10.24 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.0587914, upper bound: 318.0587914
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.0587914, upper bound: 318.0587914
time: 7.45 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1447080, upper bound: 318.1447080
time: 8.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1447080, upper bound: 318.1447080
time: 8.60 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2014316, upper bound: 318.2014316
time: 9.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2014316, upper bound: 318.2014316
time: 9.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.0981322, upper bound: 318.0981322
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.0981322, upper bound: 318.0981322
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.1613266, upper bound: 318.1613311
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.1613311, upper bound: 318.1613266
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.1868091, upper bound: 318.1868088
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.1868091, upper bound: 318.1868088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.1693169, upper bound: 318.1693169
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.1693169, upper bound: 318.1693169
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.1923117, upper bound: 318.1923115
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.1923117, upper bound: 318.1923115
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.0587914, upper bound: 318.0587914
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.0587914, upper bound: 318.0587914
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.1447080, upper bound: 318.1447080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.1447080, upper bound: 318.1447080
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.2014316, upper bound: 318.2014316
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.01
Output dim: 1, lower bound: -318.2014316, upper bound: 318.2014316

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593396
time: 8.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593396
time: 8.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593396
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593396
time: 8.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1058664, upper bound: 318.1058663
time: 8.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1058664, upper bound: 318.1058663
time: 8.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 193

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1851139, upper bound: 318.1851139
time: 9.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1851139, upper bound: 318.1851139
time: 8.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1620234, upper bound: 318.1620234
time: 9.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1620234, upper bound: 318.1620234
time: 8.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1693120, upper bound: 318.1693107
time: 9.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1693107, upper bound: 318.1693120
time: 11.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1908821, upper bound: 318.1908856
time: 9.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1908859, upper bound: 318.1908820
time: 9.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.0995750, upper bound: 318.0995792
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.0995750, upper bound: 318.0995792
time: 8.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1284623, upper bound: 318.1284623
time: 8.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1284623, upper bound: 318.1284623
time: 9.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1153219, upper bound: 318.1153218
time: 8.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1153219, upper bound: 318.1153218
time: 8.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1723419, upper bound: 318.1723392
time: 8.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1723419, upper bound: 318.1723392
time: 8.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1401222, upper bound: 318.1401229
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1401222, upper bound: 318.1401229
time: 7.29 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 15.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593396
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593396
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593396
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1058664, upper bound: 318.1058663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1058664, upper bound: 318.1058663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1851139, upper bound: 318.1851139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1851139, upper bound: 318.1851139
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1620234, upper bound: 318.1620234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1620234, upper bound: 318.1620234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1693120, upper bound: 318.1693107
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1693107, upper bound: 318.1693120
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1908821, upper bound: 318.1908856
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1908859, upper bound: 318.1908820
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.0995750, upper bound: 318.0995792
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.0995750, upper bound: 318.0995792
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1284623, upper bound: 318.1284623
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1284623, upper bound: 318.1284623
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1153219, upper bound: 318.1153218
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1153219, upper bound: 318.1153218
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1723419, upper bound: 318.1723392
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1723419, upper bound: 318.1723392
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1401222, upper bound: 318.1401229
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.84
Output dim: 1, lower bound: -318.1401222, upper bound: 318.1401229

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1592817, upper bound: 318.1592841
time: 9.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1592841, upper bound: 318.1592822
time: 9.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1593229, upper bound: 318.1593396
time: 10.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593250
time: 10.28 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 21.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 21.97
Output dim: 1, lower bound: -318.1592817, upper bound: 318.1592841
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 21.97
Output dim: 1, lower bound: -318.1592841, upper bound: 318.1592822
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 21.97
Output dim: 1, lower bound: -318.1593229, upper bound: 318.1593396
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 21.97
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593250
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593396
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1593396, upper bound: 318.1593396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1851139, upper bound: 318.1851139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1851139, upper bound: 318.1851139
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1620234, upper bound: 318.1620234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1620234, upper bound: 318.1620234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1693120, upper bound: 318.1693107
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1693107, upper bound: 318.1693120
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1908821, upper bound: 318.1908856
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1908859, upper bound: 318.1908820
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1723419, upper bound: 318.1723392
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.97
Output dim: 1, lower bound: -318.1723419, upper bound: 318.1723392
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=319.7423400878906
rel_dist={1: [-318.23529533371016, 318.23529533356407]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2244584, upper bound: 318.2244583
time: 10.94 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2244584, upper bound: 318.2244583
time: 10.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.60
Output dim: 1, lower bound: -318.2244584, upper bound: 318.2244583
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.60
Output dim: 1, lower bound: -318.2244584, upper bound: 318.2244583

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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2228783, upper bound: 318.2228779
time: 12.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2228783, upper bound: 318.2228780
time: 11.39 seconds

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2180894, upper bound: 318.2180893
time: 10.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2180894, upper bound: 318.2180894
time: 11.96 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.90 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.90
Output dim: 1, lower bound: -318.2228783, upper bound: 318.2228779
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.90
Output dim: 1, lower bound: -318.2228783, upper bound: 318.2228780
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.90
Output dim: 1, lower bound: -318.2180894, upper bound: 318.2180893
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.90
Output dim: 1, lower bound: -318.2180894, upper bound: 318.2180894

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
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 55

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1699630, upper bound: 318.1699631
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1699630, upper bound: 318.1699630
time: 10.19 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2228714, upper bound: 318.2228719
time: 11.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2228714, upper bound: 318.2228730
time: 12.95 seconds

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2158768, upper bound: 318.2158768
time: 10.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2158768, upper bound: 318.2158768
time: 11.70 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2136501, upper bound: 318.2136500
time: 9.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2136500, upper bound: 318.2136500
time: 11.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.77
Output dim: 1, lower bound: -318.1699630, upper bound: 318.1699631
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.77
Output dim: 1, lower bound: -318.1699630, upper bound: 318.1699630
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.77
Output dim: 1, lower bound: -318.2228714, upper bound: 318.2228719
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.77
Output dim: 1, lower bound: -318.2228714, upper bound: 318.2228730
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.77
Output dim: 1, lower bound: -318.2158768, upper bound: 318.2158768
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.77
Output dim: 1, lower bound: -318.2158768, upper bound: 318.2158768
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.77
Output dim: 1, lower bound: -318.2136501, upper bound: 318.2136500
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.77
Output dim: 1, lower bound: -318.2136500, upper bound: 318.2136500

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1652077, upper bound: 318.1652111
time: 9.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1652112, upper bound: 318.1652076
time: 10.09 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1627360, upper bound: 318.1627378
time: 10.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1627360, upper bound: 318.1627378
time: 9.54 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1953989, upper bound: 318.1953962
time: 11.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1953989, upper bound: 318.1953962
time: 11.10 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2228708, upper bound: 318.2228729
time: 10.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2228714, upper bound: 318.2228723
time: 11.40 seconds

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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2132722, upper bound: 318.2132756
time: 13.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2132723, upper bound: 318.2132722
time: 11.08 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2155696, upper bound: 318.2155697
time: 11.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2155697, upper bound: 318.2155696
time: 9.66 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2134649, upper bound: 318.2134648
time: 12.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2134649, upper bound: 318.2134691
time: 11.39 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2062631, upper bound: 318.2062632
time: 11.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2062631, upper bound: 318.2062632
time: 11.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.1652077, upper bound: 318.1652111
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.1652112, upper bound: 318.1652076
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.1627360, upper bound: 318.1627378
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.1627360, upper bound: 318.1627378
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.1953989, upper bound: 318.1953962
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.1953989, upper bound: 318.1953962
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.2228708, upper bound: 318.2228729
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.2228714, upper bound: 318.2228723
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.2132722, upper bound: 318.2132756
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.2132723, upper bound: 318.2132722
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.2155696, upper bound: 318.2155697
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.2155697, upper bound: 318.2155696
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.2134649, upper bound: 318.2134648
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.2134649, upper bound: 318.2134691
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.2062631, upper bound: 318.2062632
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 1, lower bound: -318.2062631, upper bound: 318.2062632

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1651195, upper bound: 318.1651214
time: 9.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1651179, upper bound: 318.1651227
time: 10.66 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1070266, upper bound: 318.1070270
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1070266, upper bound: 318.1070270
time: 8.47 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1624856, upper bound: 318.1624942
time: 11.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1624928, upper bound: 318.1624862
time: 10.58 seconds

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

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1627360, upper bound: 318.1627378
time: 10.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1627360, upper bound: 318.1627378
time: 11.02 seconds

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

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1771252, upper bound: 318.1771245
time: 10.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1771252, upper bound: 318.1771245
time: 9.14 seconds

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

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1939316, upper bound: 318.1939316
time: 12.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1939319, upper bound: 318.1939316
time: 12.29 seconds

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

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1996356, upper bound: 318.1996374
time: 10.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1996350, upper bound: 318.1996374
time: 10.80 seconds

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

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1705848, upper bound: 318.1705850
time: 9.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1705848, upper bound: 318.1705850
time: 9.77 seconds

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

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2120525, upper bound: 318.2120529
time: 11.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2120525, upper bound: 318.2120529
time: 12.75 seconds

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

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 169

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1650796, upper bound: 318.1650797
time: 9.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1650796, upper bound: 318.1650797
time: 9.28 seconds

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

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 244
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 55
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 169
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 203
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2078984, upper bound: 318.2078965
time: 11.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2078984, upper bound: 318.2078965
time: 12.29 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1651195, upper bound: 318.1651214
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1651179, upper bound: 318.1651227
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1070266, upper bound: 318.1070270
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1070266, upper bound: 318.1070270
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1624856, upper bound: 318.1624942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1624928, upper bound: 318.1624862
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1627360, upper bound: 318.1627378
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1627360, upper bound: 318.1627378
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1771252, upper bound: 318.1771245
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1771252, upper bound: 318.1771245
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1939316, upper bound: 318.1939316
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1939319, upper bound: 318.1939316
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1996356, upper bound: 318.1996374
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1996350, upper bound: 318.1996374
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1705848, upper bound: 318.1705850
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1705848, upper bound: 318.1705850
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.2120525, upper bound: 318.2120529
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.2120525, upper bound: 318.2120529
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1650796, upper bound: 318.1650797
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.1650796, upper bound: 318.1650797
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.2078984, upper bound: 318.2078965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.30
Output dim: 1, lower bound: -318.2078984, upper bound: 318.2078965
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -318.2155697, upper bound: 318.2155696
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -318.2134649, upper bound: 318.2134648
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -318.2134649, upper bound: 318.2134691
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -318.2062631, upper bound: 318.2062632
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.30
Output dim: 1, lower bound: -318.2062631, upper bound: 318.2062632
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=319.7423400878906
rel_dist={1: [-318.2352719030525, 318.23527182273534]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1854.98 seconds
