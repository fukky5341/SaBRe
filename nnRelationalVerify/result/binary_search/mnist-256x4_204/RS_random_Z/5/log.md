## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 1.84108648411
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464)
1: (-0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231)
2: (-0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522)
3: (-0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149)
4: (-0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112)
5: (-0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641)
6: (-0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580)
7: (-0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284)
8: (-0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284)
9: (-0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595)

## BASE Result
execution time: IAR + LP analysis = 1.10 + 2.90 = 4.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -1.9162456, upper bound: 1.9162456


# Binary Search by BASE starts (time budget: 2695.99 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.094357967376709
rel_dist={6: [-1.9135725282356488, 1.9135718444666665]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.094357967376709
rel_dist={6: [-1.9113006693296792, 1.9113006693296786]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.094357967376709
rel_dist={6: [-1.9066430217684205, 1.9066430217684207]}

## Binary Search Result
Binary search time: 15.44 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2680.55 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9052993, upper bound: 1.9052935
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9052935, upper bound: 1.9052993
time: 2.17 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.02
Output dim: 6, lower bound: -1.9052993, upper bound: 1.9052935
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.02
Output dim: 6, lower bound: -1.9052935, upper bound: 1.9052993

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8983957, upper bound: 1.8982640
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8982640, upper bound: 1.8983832
time: 2.00 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8190154, upper bound: 1.8190154
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8190154, upper bound: 1.8190154
time: 1.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 6, lower bound: -1.8983957, upper bound: 1.8982640
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 6, lower bound: -1.8982640, upper bound: 1.8983832
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 4.32
Output dim: 6, lower bound: -1.8190154, upper bound: 1.8190154
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 4.32
Output dim: 6, lower bound: -1.8190154, upper bound: 1.8190154

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8964349, upper bound: 1.8963112
time: 2.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8964349, upper bound: 1.8963113
time: 2.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7298867, upper bound: 1.7299231
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7298867, upper bound: 1.7299231
time: 1.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.64 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 6, lower bound: -1.8964349, upper bound: 1.8963112
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 6, lower bound: -1.8964349, upper bound: 1.8963113
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 6, lower bound: -1.7298867, upper bound: 1.7299231
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 6, lower bound: -1.7298867, upper bound: 1.7299231

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8729785, upper bound: 1.8728705
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8729785, upper bound: 1.8728705
time: 2.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8955133, upper bound: 1.8953996
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8955260, upper bound: 1.8953981
time: 2.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.56
Output dim: 6, lower bound: -1.8729785, upper bound: 1.8728705
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.56
Output dim: 6, lower bound: -1.8729785, upper bound: 1.8728705
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.56
Output dim: 6, lower bound: -1.8955133, upper bound: 1.8953996
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.56
Output dim: 6, lower bound: -1.8955260, upper bound: 1.8953981

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8715343, upper bound: 1.8714508
time: 2.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8715343, upper bound: 1.8714508
time: 2.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8727436, upper bound: 1.8726131
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8727114, upper bound: 1.8726382
time: 2.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8859455, upper bound: 1.8858551
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8859455, upper bound: 1.8858551
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712780, upper bound: 1.8711680
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712780, upper bound: 1.8711680
time: 2.18 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 6, lower bound: -1.8715343, upper bound: 1.8714508
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 6, lower bound: -1.8715343, upper bound: 1.8714508
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 6, lower bound: -1.8727436, upper bound: 1.8726131
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 6, lower bound: -1.8727114, upper bound: 1.8726382
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 6, lower bound: -1.8859455, upper bound: 1.8858551
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 6, lower bound: -1.8859455, upper bound: 1.8858551
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 6, lower bound: -1.8712780, upper bound: 1.8711680
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.37
Output dim: 6, lower bound: -1.8712780, upper bound: 1.8711680

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712834, upper bound: 1.8711616
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712415, upper bound: 1.8711935
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712834, upper bound: 1.8711616
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712415, upper bound: 1.8711936
time: 2.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8691971, upper bound: 1.8690594
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8691971, upper bound: 1.8690594
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712664, upper bound: 1.8712140
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712664, upper bound: 1.8712140
time: 2.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8859455, upper bound: 1.8857033
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8857364, upper bound: 1.8858551
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7847581, upper bound: 1.7846517
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7847581, upper bound: 1.7846517
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8682463, upper bound: 1.8681554
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8682463, upper bound: 1.8681554
time: 2.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8524079, upper bound: 1.8522987
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8524067, upper bound: 1.8523002
time: 2.12 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8712834, upper bound: 1.8711616
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8712415, upper bound: 1.8711935
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8712834, upper bound: 1.8711616
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8712415, upper bound: 1.8711936
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8691971, upper bound: 1.8690594
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8691971, upper bound: 1.8690594
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8712664, upper bound: 1.8712140
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8712664, upper bound: 1.8712140
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8859455, upper bound: 1.8857033
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8857364, upper bound: 1.8858551
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.7847581, upper bound: 1.7846517
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.7847581, upper bound: 1.7846517
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8682463, upper bound: 1.8681554
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8682463, upper bound: 1.8681554
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8524079, upper bound: 1.8522987
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.40
Output dim: 6, lower bound: -1.8524067, upper bound: 1.8523002

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712834, upper bound: 1.8711302
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712381, upper bound: 1.8711616
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8683608, upper bound: 1.8683071
time: 8.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8683447, upper bound: 1.8683121
time: 2.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7600869, upper bound: 1.7600143
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7600869, upper bound: 1.7600143
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8712374, upper bound: 1.8711405
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8711688, upper bound: 1.8711892
time: 2.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8444323, upper bound: 1.8442716
time: 2.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8444312, upper bound: 1.8442755
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8584780, upper bound: 1.8583251
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8584779, upper bound: 1.8583245
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7717004, upper bound: 1.7717119
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7717004, upper bound: 1.7717119
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8633450, upper bound: 1.8632938
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8633450, upper bound: 1.8632938
time: 2.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8854969, upper bound: 1.8853301
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8855679, upper bound: 1.8852382
time: 2.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8856514, upper bound: 1.8857733
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8856514, upper bound: 1.8857733
time: 2.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8682463, upper bound: 1.8679088
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8679960, upper bound: 1.8681554
time: 2.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6782094, upper bound: 1.6782069
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6782094, upper bound: 1.6782069
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8524079, upper bound: 1.8522721
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8523922, upper bound: 1.8522987
time: 2.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8513892, upper bound: 1.8512795
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8513892, upper bound: 1.8512795
time: 1.94 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8712834, upper bound: 1.8711302
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8712381, upper bound: 1.8711616
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8683608, upper bound: 1.8683071
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8683447, upper bound: 1.8683121
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.7600869, upper bound: 1.7600143
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.7600869, upper bound: 1.7600143
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8712374, upper bound: 1.8711405
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8711688, upper bound: 1.8711892
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8444323, upper bound: 1.8442716
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8444312, upper bound: 1.8442755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8584780, upper bound: 1.8583251
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8584779, upper bound: 1.8583245
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.7717004, upper bound: 1.7717119
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.7717004, upper bound: 1.7717119
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8633450, upper bound: 1.8632938
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8633450, upper bound: 1.8632938
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8854969, upper bound: 1.8853301
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8855679, upper bound: 1.8852382
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8856514, upper bound: 1.8857733
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8856514, upper bound: 1.8857733
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8682463, upper bound: 1.8679088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8679960, upper bound: 1.8681554
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.6782094, upper bound: 1.6782069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.6782094, upper bound: 1.6782069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8524079, upper bound: 1.8522721
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8523922, upper bound: 1.8522987
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8513892, upper bound: 1.8512795
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.15
Output dim: 6, lower bound: -1.8513892, upper bound: 1.8512795

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8233210, upper bound: 1.8232508
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8233210, upper bound: 1.8232508
time: 2.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7992713, upper bound: 1.7991716
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7992713, upper bound: 1.7991716
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8563776, upper bound: 1.8563416
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8563776, upper bound: 1.8563416
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8678979, upper bound: 1.8679338
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8679591, upper bound: 1.8678824
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8710582, upper bound: 1.8708216
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8709239, upper bound: 1.8709624
time: 2.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454393, upper bound: 1.8454257
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454388, upper bound: 1.8454263
time: 2.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7892279, upper bound: 1.7891613
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7892279, upper bound: 1.7891613
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8444312, upper bound: 1.8442641
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8443999, upper bound: 1.8442755
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8575955, upper bound: 1.8575029
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8576546, upper bound: 1.8574503
time: 2.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496374, upper bound: 1.8495271
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496374, upper bound: 1.8495271
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8166350, upper bound: 1.8166132
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8166350, upper bound: 1.8166132
time: 2.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8166350, upper bound: 1.8166132
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8166350, upper bound: 1.8166132
time: 2.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457236, upper bound: 1.8456032
time: 2.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457236, upper bound: 1.8456032
time: 2.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8465188, upper bound: 1.8461911
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8465188, upper bound: 1.8461911
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8851597, upper bound: 1.8853525
time: 3.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8852392, upper bound: 1.8852625
time: 2.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8461531, upper bound: 1.8462139
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8461531, upper bound: 1.8462138
time: 2.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7392630, upper bound: 1.7392675
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7392630, upper bound: 1.7392675
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7215918, upper bound: 1.7216234
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7215918, upper bound: 1.7216234
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7364953, upper bound: 1.7364740
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7364953, upper bound: 1.7364740
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522936, upper bound: 1.8521756
time: 2.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8522524, upper bound: 1.8521989
time: 2.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7820588, upper bound: 1.7820038
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7820588, upper bound: 1.7820038
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8393438, upper bound: 1.8392297
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8393261, upper bound: 1.8392446
time: 2.22 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 5.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8233210, upper bound: 1.8232508
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8233210, upper bound: 1.8232508
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7992713, upper bound: 1.7991716
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7992713, upper bound: 1.7991716
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8563776, upper bound: 1.8563416
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8563776, upper bound: 1.8563416
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8678979, upper bound: 1.8679338
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8679591, upper bound: 1.8678824
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8710582, upper bound: 1.8708216
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8709239, upper bound: 1.8709624
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8454393, upper bound: 1.8454257
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8454388, upper bound: 1.8454263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7892279, upper bound: 1.7891613
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7892279, upper bound: 1.7891613
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8444312, upper bound: 1.8442641
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8443999, upper bound: 1.8442755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8575955, upper bound: 1.8575029
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8576546, upper bound: 1.8574503
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8496374, upper bound: 1.8495271
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8496374, upper bound: 1.8495271
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8166350, upper bound: 1.8166132
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8166350, upper bound: 1.8166132
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8166350, upper bound: 1.8166132
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8166350, upper bound: 1.8166132
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8457236, upper bound: 1.8456032
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8457236, upper bound: 1.8456032
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8465188, upper bound: 1.8461911
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8465188, upper bound: 1.8461911
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8851597, upper bound: 1.8853525
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8852392, upper bound: 1.8852625
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8461531, upper bound: 1.8462139
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8461531, upper bound: 1.8462138
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7392630, upper bound: 1.7392675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7392630, upper bound: 1.7392675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7215918, upper bound: 1.7216234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7215918, upper bound: 1.7216234
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7364953, upper bound: 1.7364740
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7364953, upper bound: 1.7364740
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8522936, upper bound: 1.8521756
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8522524, upper bound: 1.8521989
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7820588, upper bound: 1.7820038
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.7820588, upper bound: 1.7820038
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8393438, upper bound: 1.8392297
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 5.58
Output dim: 6, lower bound: -1.8393261, upper bound: 1.8392446

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8144557, upper bound: 1.8144217
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8144557, upper bound: 1.8144217
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8140189, upper bound: 1.8140044
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8140189, upper bound: 1.8140044
time: 2.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7957586, upper bound: 1.7957202
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7957586, upper bound: 1.7957202
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7563052, upper bound: 1.7563231
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7563052, upper bound: 1.7563231
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8677345, upper bound: 1.8675067
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8677345, upper bound: 1.8675067
time: 2.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8302202, upper bound: 1.8302208
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8302202, upper bound: 1.8302208
time: 2.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454393, upper bound: 1.8453470
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452546, upper bound: 1.8454257
time: 2.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454388, upper bound: 1.8451062
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8451105, upper bound: 1.8454263
time: 2.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 207

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8442861, upper bound: 1.8440204
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441319, upper bound: 1.8441152
time: 2.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8443999, upper bound: 1.8440341
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8441273, upper bound: 1.8442755
time: 1.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7760251, upper bound: 1.7759931
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7760251, upper bound: 1.7759931
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8488182, upper bound: 1.8486471
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8488182, upper bound: 1.8486471
time: 2.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8494690, upper bound: 1.8492274
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493425, upper bound: 1.8493641
time: 2.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496374, upper bound: 1.8495255
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496225, upper bound: 1.8495271
time: 2.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454338, upper bound: 1.8453057
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454035, upper bound: 1.8453285
time: 2.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8430920, upper bound: 1.8429852
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8430920, upper bound: 1.8429860
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8436387, upper bound: 1.8433143
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8436209, upper bound: 1.8433159
time: 2.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8463302, upper bound: 1.8458360
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8461067, upper bound: 1.8459918
time: 2.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8850231, upper bound: 1.8851838
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8849740, upper bound: 1.8852267
time: 2.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8582144, upper bound: 1.8582379
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8582144, upper bound: 1.8582379
time: 2.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8459924, upper bound: 1.8459228
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8458514, upper bound: 1.8460434
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7326342, upper bound: 1.7326446
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7326342, upper bound: 1.7326446
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7819689, upper bound: 1.7818993
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7819689, upper bound: 1.7818993
time: 2.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8516339, upper bound: 1.8516531
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8517134, upper bound: 1.8515665
time: 2.21 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 5.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8144557, upper bound: 1.8144217
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8144557, upper bound: 1.8144217
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8140189, upper bound: 1.8140044
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8140189, upper bound: 1.8140044
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.7957586, upper bound: 1.7957202
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.7957586, upper bound: 1.7957202
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.7563052, upper bound: 1.7563231
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.7563052, upper bound: 1.7563231
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8677345, upper bound: 1.8675067
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8677345, upper bound: 1.8675067
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8302202, upper bound: 1.8302208
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8302202, upper bound: 1.8302208
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8454393, upper bound: 1.8453470
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8452546, upper bound: 1.8454257
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8454388, upper bound: 1.8451062
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8451105, upper bound: 1.8454263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8442861, upper bound: 1.8440204
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8441319, upper bound: 1.8441152
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8443999, upper bound: 1.8440341
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8441273, upper bound: 1.8442755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.7760251, upper bound: 1.7759931
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.7760251, upper bound: 1.7759931
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8488182, upper bound: 1.8486471
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8488182, upper bound: 1.8486471
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8494690, upper bound: 1.8492274
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8493425, upper bound: 1.8493641
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8496374, upper bound: 1.8495255
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8496225, upper bound: 1.8495271
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8454338, upper bound: 1.8453057
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8454035, upper bound: 1.8453285
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8430920, upper bound: 1.8429852
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8430920, upper bound: 1.8429860
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8436387, upper bound: 1.8433143
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8436209, upper bound: 1.8433159
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8463302, upper bound: 1.8458360
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8461067, upper bound: 1.8459918
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8850231, upper bound: 1.8851838
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8849740, upper bound: 1.8852267
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8582144, upper bound: 1.8582379
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8582144, upper bound: 1.8582379
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8459924, upper bound: 1.8459228
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8458514, upper bound: 1.8460434
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.7326342, upper bound: 1.7326446
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.7326342, upper bound: 1.7326446
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.7819689, upper bound: 1.7818993
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.7819689, upper bound: 1.7818993
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8516339, upper bound: 1.8516531
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.40
Output dim: 6, lower bound: -1.8517134, upper bound: 1.8515665

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8650161, upper bound: 1.8647736
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8649987, upper bound: 1.8647820
time: 2.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8592629, upper bound: 1.8590365
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8592629, upper bound: 1.8590365
time: 2.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7803019, upper bound: 1.7802498
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7803019, upper bound: 1.7802498
time: 1.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452523, upper bound: 1.8453883
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8452278, upper bound: 1.8454257
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7696585, upper bound: 1.7696544
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7696585, upper bound: 1.7696544
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8383031, upper bound: 1.8385829
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8383031, upper bound: 1.8385829
time: 2.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7773733, upper bound: 1.7771957
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7773733, upper bound: 1.7771957
time: 2.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7889560, upper bound: 1.7889684
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7889560, upper bound: 1.7889684
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8036044, upper bound: 1.8034218
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8036044, upper bound: 1.8034218
time: 2.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8337848, upper bound: 1.8338869
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8337877, upper bound: 1.8338856
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8481077, upper bound: 1.8479571
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8481077, upper bound: 1.8479571
time: 2.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8478965, upper bound: 1.8477476
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8479173, upper bound: 1.8477326
time: 2.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493232, upper bound: 1.8489827
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8491874, upper bound: 1.8490772
time: 2.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8493396, upper bound: 1.8493155
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8492754, upper bound: 1.8493604
time: 1.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8399179, upper bound: 1.8398150
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8399179, upper bound: 1.8398150
time: 2.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8106258, upper bound: 1.8105273
time: 3.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8106258, upper bound: 1.8105273
time: 3.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439202, upper bound: 1.8437657
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8439202, upper bound: 1.8437657
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6547023, upper bound: 1.6547030
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6547023, upper bound: 1.6547030
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8430920, upper bound: 1.8429659
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8430732, upper bound: 1.8429852
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8292754, upper bound: 1.8291567
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8292679, upper bound: 1.8291875
time: 2.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8436387, upper bound: 1.8432930
time: 2.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8436150, upper bound: 1.8433027
time: 2.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407845, upper bound: 1.8404806
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8407753, upper bound: 1.8404811
time: 2.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8218012, upper bound: 1.8214657
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8218012, upper bound: 1.8214657
time: 2.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8432388, upper bound: 1.8431175
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8431988, upper bound: 1.8431195
time: 2.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8846990, upper bound: 1.8848098
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8846344, upper bound: 1.8848557
time: 2.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453084, upper bound: 1.8454688
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8453084, upper bound: 1.8454688
time: 2.41 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 5.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8650161, upper bound: 1.8647736
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8649987, upper bound: 1.8647820
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8592629, upper bound: 1.8590365
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8592629, upper bound: 1.8590365
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.7803019, upper bound: 1.7802498
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.7803019, upper bound: 1.7802498
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8452523, upper bound: 1.8453883
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8452278, upper bound: 1.8454257
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.7696585, upper bound: 1.7696544
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.7696585, upper bound: 1.7696544
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8383031, upper bound: 1.8385829
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8383031, upper bound: 1.8385829
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.7773733, upper bound: 1.7771957
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.7773733, upper bound: 1.7771957
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.7889560, upper bound: 1.7889684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.7889560, upper bound: 1.7889684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8036044, upper bound: 1.8034218
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8036044, upper bound: 1.8034218
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8337848, upper bound: 1.8338869
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8337877, upper bound: 1.8338856
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8481077, upper bound: 1.8479571
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8481077, upper bound: 1.8479571
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8478965, upper bound: 1.8477476
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8479173, upper bound: 1.8477326
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8493232, upper bound: 1.8489827
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8491874, upper bound: 1.8490772
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8493396, upper bound: 1.8493155
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8492754, upper bound: 1.8493604
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8399179, upper bound: 1.8398150
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8399179, upper bound: 1.8398150
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8106258, upper bound: 1.8105273
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8106258, upper bound: 1.8105273
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8439202, upper bound: 1.8437657
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8439202, upper bound: 1.8437657
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.6547023, upper bound: 1.6547030
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.6547023, upper bound: 1.6547030
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8430920, upper bound: 1.8429659
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8430732, upper bound: 1.8429852
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8292754, upper bound: 1.8291567
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8292679, upper bound: 1.8291875
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8436387, upper bound: 1.8432930
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8436150, upper bound: 1.8433027
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8407845, upper bound: 1.8404806
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8407753, upper bound: 1.8404811
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8218012, upper bound: 1.8214657
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8218012, upper bound: 1.8214657
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8432388, upper bound: 1.8431175
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8431988, upper bound: 1.8431195
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8846990, upper bound: 1.8848098
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8846344, upper bound: 1.8848557
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8453084, upper bound: 1.8454688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.93
Output dim: 6, lower bound: -1.8453084, upper bound: 1.8454688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.93
Output dim: 6, lower bound: -1.8582144, upper bound: 1.8582379
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.93
Output dim: 6, lower bound: -1.8582144, upper bound: 1.8582379
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.93
Output dim: 6, lower bound: -1.8459924, upper bound: 1.8459228
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.93
Output dim: 6, lower bound: -1.8458514, upper bound: 1.8460434
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.93
Output dim: 6, lower bound: -1.8516339, upper bound: 1.8516531
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.93
Output dim: 6, lower bound: -1.8517134, upper bound: 1.8515665
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.094357967376709
rel_dist={6: [-1.9135725282356488, 1.9135718444666665]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9113007, upper bound: 1.9112229
time: 2.90 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9112229, upper bound: 1.9113007
time: 2.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.34 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.34
Output dim: 6, lower bound: -1.9113007, upper bound: 1.9112229
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.34
Output dim: 6, lower bound: -1.9112229, upper bound: 1.9113007

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8391388, upper bound: 1.8391171
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8391388, upper bound: 1.8391171
time: 1.84 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9101999, upper bound: 1.9102805
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9102007, upper bound: 1.9102785
time: 2.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.87 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 5.87
Output dim: 6, lower bound: -1.8391388, upper bound: 1.8391171
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 5.87
Output dim: 6, lower bound: -1.8391388, upper bound: 1.8391171
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.87
Output dim: 6, lower bound: -1.9101999, upper bound: 1.9102805
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.87
Output dim: 6, lower bound: -1.9102007, upper bound: 1.9102785

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9101999, upper bound: 1.9102439
time: 2.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9101605, upper bound: 1.9102805
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8230910, upper bound: 1.8231129
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8230910, upper bound: 1.8231129
time: 2.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.22 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.22
Output dim: 6, lower bound: -1.9101999, upper bound: 1.9102439
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.22
Output dim: 6, lower bound: -1.9101605, upper bound: 1.9102805
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 5.22
Output dim: 6, lower bound: -1.8230910, upper bound: 1.8231129
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 5.22
Output dim: 6, lower bound: -1.8230910, upper bound: 1.8231129

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6777494, upper bound: 1.6777452
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6777494, upper bound: 1.6777452
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7582014, upper bound: 1.7582359
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7582014, upper bound: 1.7582359
time: 1.85 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.73 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 6, lower bound: -1.6777494, upper bound: 1.6777452
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 6, lower bound: -1.6777494, upper bound: 1.6777452
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 6, lower bound: -1.7582014, upper bound: 1.7582359
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.73
Output dim: 6, lower bound: -1.7582014, upper bound: 1.7582359
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.094357967376709
rel_dist={6: [-1.9113006693296792, 1.9113006693296786]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9121802, upper bound: 1.9116467
time: 2.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9116467, upper bound: 1.9121803
time: 2.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.71 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.71
Output dim: 6, lower bound: -1.9121802, upper bound: 1.9116467
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.71
Output dim: 6, lower bound: -1.9116467, upper bound: 1.9121803

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9025280, upper bound: 1.9020225
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9025280, upper bound: 1.9020225
time: 2.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8864247, upper bound: 1.8867719
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8864247, upper bound: 1.8867719
time: 2.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.37 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 6, lower bound: -1.9025280, upper bound: 1.9020225
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 6, lower bound: -1.9025280, upper bound: 1.9020225
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 6, lower bound: -1.8864247, upper bound: 1.8867719
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 6, lower bound: -1.8864247, upper bound: 1.8867719

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6576606, upper bound: 1.6576213
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6576606, upper bound: 1.6576213
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9024545, upper bound: 1.9018985
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9023872, upper bound: 1.9019466
time: 2.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8526708, upper bound: 1.8529210
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8526708, upper bound: 1.8529210
time: 2.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8393936, upper bound: 1.8396526
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8393936, upper bound: 1.8396526
time: 1.94 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.91 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.91
Output dim: 6, lower bound: -1.6576606, upper bound: 1.6576213
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.91
Output dim: 6, lower bound: -1.6576606, upper bound: 1.6576213
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.91
Output dim: 6, lower bound: -1.9024545, upper bound: 1.9018985
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.91
Output dim: 6, lower bound: -1.9023872, upper bound: 1.9019466
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.91
Output dim: 6, lower bound: -1.8526708, upper bound: 1.8529210
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.91
Output dim: 6, lower bound: -1.8526708, upper bound: 1.8529210
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 4.91
Output dim: 6, lower bound: -1.8393936, upper bound: 1.8396526
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 4.91
Output dim: 6, lower bound: -1.8393936, upper bound: 1.8396526

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9023735, upper bound: 1.9018597
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9024123, upper bound: 1.9017957
time: 2.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8618792, upper bound: 1.8616362
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8618792, upper bound: 1.8616362
time: 2.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8491309, upper bound: 1.8493857
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8491309, upper bound: 1.8493857
time: 1.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8516385, upper bound: 1.8519015
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8516428, upper bound: 1.8519012
time: 2.16 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.26 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 6, lower bound: -1.9023735, upper bound: 1.9018597
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 6, lower bound: -1.9024123, upper bound: 1.9017957
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 6, lower bound: -1.8618792, upper bound: 1.8616362
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 6, lower bound: -1.8618792, upper bound: 1.8616362
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 6, lower bound: -1.8491309, upper bound: 1.8493857
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 6, lower bound: -1.8491309, upper bound: 1.8493857
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 6, lower bound: -1.8516385, upper bound: 1.8519015
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 6, lower bound: -1.8516428, upper bound: 1.8519012

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9023699, upper bound: 1.9018451
time: 2.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.9023686, upper bound: 1.9018470
time: 2.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6568572, upper bound: 1.6568046
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6568572, upper bound: 1.6568046
time: 1.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8618792, upper bound: 1.8616327
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8618580, upper bound: 1.8616362
time: 2.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8618792, upper bound: 1.8616301
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8618656, upper bound: 1.8616362
time: 2.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8461307, upper bound: 1.8463744
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8461295, upper bound: 1.8463892
time: 1.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8491252, upper bound: 1.8493416
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8490967, upper bound: 1.8493857
time: 2.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8487327, upper bound: 1.8489871
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8487307, upper bound: 1.8489871
time: 2.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8514619, upper bound: 1.8517135
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8514291, upper bound: 1.8517371
time: 2.12 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.83 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.9023699, upper bound: 1.9018451
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.9023686, upper bound: 1.9018470
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.6568572, upper bound: 1.6568046
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.6568572, upper bound: 1.6568046
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8618792, upper bound: 1.8616327
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8618580, upper bound: 1.8616362
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8618792, upper bound: 1.8616301
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8618656, upper bound: 1.8616362
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8461307, upper bound: 1.8463744
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8461295, upper bound: 1.8463892
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8491252, upper bound: 1.8493416
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8490967, upper bound: 1.8493857
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8487327, upper bound: 1.8489871
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8487307, upper bound: 1.8489871
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8514619, upper bound: 1.8517135
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 6, lower bound: -1.8514291, upper bound: 1.8517371

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8092045, upper bound: 1.8091042
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8092045, upper bound: 1.8091042
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8818339, upper bound: 1.8814993
time: 2.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8818321, upper bound: 1.8814993
time: 2.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496121, upper bound: 1.8493513
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496121, upper bound: 1.8493513
time: 2.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6632867, upper bound: 1.6633059
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6632867, upper bound: 1.6633059
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7465573, upper bound: 1.7465395
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7465573, upper bound: 1.7465395
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8607782, upper bound: 1.8605490
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8607783, upper bound: 1.8605493
time: 2.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8460867, upper bound: 1.8462710
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8460193, upper bound: 1.8463645
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8461243, upper bound: 1.8463488
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8460913, upper bound: 1.8463892
time: 1.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8490281, upper bound: 1.8492380
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8490275, upper bound: 1.8492502
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8460915, upper bound: 1.8463744
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8460921, upper bound: 1.8463892
time: 2.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8100540, upper bound: 1.8101901
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8100540, upper bound: 1.8101901
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8131870, upper bound: 1.8133686
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8131870, upper bound: 1.8133686
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8427220, upper bound: 1.8429793
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8427220, upper bound: 1.8429793
time: 2.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8485135, upper bound: 1.8487633
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484930, upper bound: 1.8488014
time: 2.03 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.96 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8092045, upper bound: 1.8091042
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8092045, upper bound: 1.8091042
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8818339, upper bound: 1.8814993
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8818321, upper bound: 1.8814993
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8496121, upper bound: 1.8493513
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8496121, upper bound: 1.8493513
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.6632867, upper bound: 1.6633059
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.6632867, upper bound: 1.6633059
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.7465573, upper bound: 1.7465395
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.7465573, upper bound: 1.7465395
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8607782, upper bound: 1.8605490
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8607783, upper bound: 1.8605493
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8460867, upper bound: 1.8462710
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8460193, upper bound: 1.8463645
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8461243, upper bound: 1.8463488
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8460913, upper bound: 1.8463892
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8490281, upper bound: 1.8492380
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8490275, upper bound: 1.8492502
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8460915, upper bound: 1.8463744
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8460921, upper bound: 1.8463892
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8100540, upper bound: 1.8101901
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8100540, upper bound: 1.8101901
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8131870, upper bound: 1.8133686
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8131870, upper bound: 1.8133686
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8427220, upper bound: 1.8429793
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8427220, upper bound: 1.8429793
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8485135, upper bound: 1.8487633
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -1.8484930, upper bound: 1.8488014

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8807284, upper bound: 1.8803924
time: 2.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8807290, upper bound: 1.8803732
time: 2.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8043458, upper bound: 1.8042059
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8043458, upper bound: 1.8042059
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8469210, upper bound: 1.8466511
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8468922, upper bound: 1.8466717
time: 2.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8272889, upper bound: 1.8271105
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8272889, upper bound: 1.8271105
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8586675, upper bound: 1.8584386
time: 2.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8586653, upper bound: 1.8584387
time: 2.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8605992, upper bound: 1.8602686
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8604800, upper bound: 1.8603641
time: 2.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8425512, upper bound: 1.8427121
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8425498, upper bound: 1.8427121
time: 2.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7553476, upper bound: 1.7554024
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7553476, upper bound: 1.7554024
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7839823, upper bound: 1.7840537
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7839823, upper bound: 1.7840537
time: 2.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6694940, upper bound: 1.6694954
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6694940, upper bound: 1.6694954
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8490281, upper bound: 1.8492184
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8490152, upper bound: 1.8492380
time: 2.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7891963, upper bound: 1.7893119
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7891963, upper bound: 1.7893119
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8434093, upper bound: 1.8436812
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8434096, upper bound: 1.8436809
time: 2.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8459062, upper bound: 1.8462182
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8458950, upper bound: 1.8462297
time: 2.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8425008, upper bound: 1.8426487
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8424409, upper bound: 1.8427446
time: 3.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8397115, upper bound: 1.8399173
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8396769, upper bound: 1.8399363
time: 2.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8485135, upper bound: 1.8485696
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8483431, upper bound: 1.8487633
time: 2.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8204323, upper bound: 1.8206280
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8204323, upper bound: 1.8206280
time: 2.27 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 7.70 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8807284, upper bound: 1.8803924
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8807290, upper bound: 1.8803732
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8043458, upper bound: 1.8042059
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8043458, upper bound: 1.8042059
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8469210, upper bound: 1.8466511
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8468922, upper bound: 1.8466717
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8272889, upper bound: 1.8271105
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8272889, upper bound: 1.8271105
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8586675, upper bound: 1.8584386
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8586653, upper bound: 1.8584387
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8605992, upper bound: 1.8602686
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8604800, upper bound: 1.8603641
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8425512, upper bound: 1.8427121
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8425498, upper bound: 1.8427121
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.7553476, upper bound: 1.7554024
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.7553476, upper bound: 1.7554024
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.7839823, upper bound: 1.7840537
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.7839823, upper bound: 1.7840537
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.6694940, upper bound: 1.6694954
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.6694940, upper bound: 1.6694954
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8490281, upper bound: 1.8492184
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8490152, upper bound: 1.8492380
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.7891963, upper bound: 1.7893119
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.7891963, upper bound: 1.7893119
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8434093, upper bound: 1.8436812
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8434096, upper bound: 1.8436809
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8459062, upper bound: 1.8462182
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8458950, upper bound: 1.8462297
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8425008, upper bound: 1.8426487
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8424409, upper bound: 1.8427446
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8397115, upper bound: 1.8399173
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8396769, upper bound: 1.8399363
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8485135, upper bound: 1.8485696
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8483431, upper bound: 1.8487633
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8204323, upper bound: 1.8206280
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 6, lower bound: -1.8204323, upper bound: 1.8206280

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8805490, upper bound: 1.8801345
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8804452, upper bound: 1.8802273
time: 2.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8805861, upper bound: 1.8802290
time: 2.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8805204, upper bound: 1.8802299
time: 2.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6834224, upper bound: 1.6834176
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6834224, upper bound: 1.6834176
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8466959, upper bound: 1.8464376
time: 2.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8466149, upper bound: 1.8464848
time: 2.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8466044, upper bound: 1.8463689
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8466046, upper bound: 1.8463653
time: 2.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7432083, upper bound: 1.7432060
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7432083, upper bound: 1.7432060
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8350247, upper bound: 1.8347578
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8350247, upper bound: 1.8347578
time: 2.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496915, upper bound: 1.8494955
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496468, upper bound: 1.8495638
time: 2.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8425419, upper bound: 1.8426670
time: 2.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8425176, upper bound: 1.8427101
time: 2.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8292346, upper bound: 1.8293622
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8292353, upper bound: 1.8293610
time: 2.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8461437, upper bound: 1.8463327
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8461435, upper bound: 1.8463329
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8198429, upper bound: 1.8199862
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8198429, upper bound: 1.8199862
time: 2.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8406998, upper bound: 1.8409506
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8406998, upper bound: 1.8409507
time: 2.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7234179, upper bound: 1.7234368
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7234179, upper bound: 1.7234368
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8459007, upper bound: 1.8461738
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8458663, upper bound: 1.8462182
time: 2.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8319999, upper bound: 1.8322837
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8320030, upper bound: 1.8322871
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8423081, upper bound: 1.8424276
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8423027, upper bound: 1.8424657
time: 2.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7186298, upper bound: 1.7186998
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7186298, upper bound: 1.7186998
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8485135, upper bound: 1.8485342
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8485100, upper bound: 1.8485588
time: 2.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7961337, upper bound: 1.7963447
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7961337, upper bound: 1.7963447
time: 1.53 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.15 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8805490, upper bound: 1.8801345
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8804452, upper bound: 1.8802273
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8805861, upper bound: 1.8802290
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8805204, upper bound: 1.8802299
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.6834224, upper bound: 1.6834176
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.6834224, upper bound: 1.6834176
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8466959, upper bound: 1.8464376
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8466149, upper bound: 1.8464848
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8466044, upper bound: 1.8463689
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8466046, upper bound: 1.8463653
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.7432083, upper bound: 1.7432060
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.7432083, upper bound: 1.7432060
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8350247, upper bound: 1.8347578
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8350247, upper bound: 1.8347578
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8496915, upper bound: 1.8494955
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8496468, upper bound: 1.8495638
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8425419, upper bound: 1.8426670
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8425176, upper bound: 1.8427101
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8292346, upper bound: 1.8293622
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8292353, upper bound: 1.8293610
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8461437, upper bound: 1.8463327
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8461435, upper bound: 1.8463329
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8198429, upper bound: 1.8199862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8198429, upper bound: 1.8199862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8406998, upper bound: 1.8409506
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8406998, upper bound: 1.8409507
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.7234179, upper bound: 1.7234368
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.7234179, upper bound: 1.7234368
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8459007, upper bound: 1.8461738
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8458663, upper bound: 1.8462182
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8319999, upper bound: 1.8322837
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8320030, upper bound: 1.8322871
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8423081, upper bound: 1.8424276
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8423027, upper bound: 1.8424657
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.7186298, upper bound: 1.7186998
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.7186298, upper bound: 1.7186998
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8485135, upper bound: 1.8485342
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.8485100, upper bound: 1.8485588
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.7961337, upper bound: 1.7963447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.15
Output dim: 6, lower bound: -1.7961337, upper bound: 1.7963447

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8804063, upper bound: 1.8799904
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8803316, upper bound: 1.8799931
time: 2.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8774444, upper bound: 1.8772223
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8774444, upper bound: 1.8772223
time: 3.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6136191, upper bound: 1.6135458
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6136191, upper bound: 1.6135458
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8782439, upper bound: 1.8779568
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8782410, upper bound: 1.8779568
time: 2.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8007703, upper bound: 1.8006555
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8007703, upper bound: 1.8006555
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454410, upper bound: 1.8453318
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454410, upper bound: 1.8453318
time: 2.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454678, upper bound: 1.8452379
time: 2.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454678, upper bound: 1.8452379
time: 2.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8466046, upper bound: 1.8463594
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8465828, upper bound: 1.8463653
time: 1.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8037143, upper bound: 1.8036116
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8037143, upper bound: 1.8036116
time: 1.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8496418, upper bound: 1.8494924
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8495774, upper bound: 1.8495581
time: 2.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8292341, upper bound: 1.8293023
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8292345, upper bound: 1.8293000
time: 3.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8425025, upper bound: 1.8425272
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8422916, upper bound: 1.8427101
time: 2.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6718710, upper bound: 1.6718610
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6718710, upper bound: 1.6718610
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8060098, upper bound: 1.8060946
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8060098, upper bound: 1.8060946
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8356677, upper bound: 1.8358020
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8356076, upper bound: 1.8358638
time: 1.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8458158, upper bound: 1.8460647
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8457464, upper bound: 1.8462022
time: 1.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8422709, upper bound: 1.8423601
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8422271, upper bound: 1.8424276
time: 2.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8296177, upper bound: 1.8296560
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8296177, upper bound: 1.8296560
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7887112, upper bound: 1.7887151
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7887112, upper bound: 1.7887151
time: 1.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484133, upper bound: 1.8484596
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8484131, upper bound: 1.8484752
time: 2.35 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 5.90 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8804063, upper bound: 1.8799904
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8803316, upper bound: 1.8799931
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8774444, upper bound: 1.8772223
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8774444, upper bound: 1.8772223
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.6136191, upper bound: 1.6135458
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.6136191, upper bound: 1.6135458
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8782439, upper bound: 1.8779568
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8782410, upper bound: 1.8779568
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8007703, upper bound: 1.8006555
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8007703, upper bound: 1.8006555
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8454410, upper bound: 1.8453318
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8454410, upper bound: 1.8453318
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8454678, upper bound: 1.8452379
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8454678, upper bound: 1.8452379
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8466046, upper bound: 1.8463594
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8465828, upper bound: 1.8463653
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8037143, upper bound: 1.8036116
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8037143, upper bound: 1.8036116
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8496418, upper bound: 1.8494924
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8495774, upper bound: 1.8495581
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8292341, upper bound: 1.8293023
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8292345, upper bound: 1.8293000
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8425025, upper bound: 1.8425272
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8422916, upper bound: 1.8427101
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.6718710, upper bound: 1.6718610
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.6718710, upper bound: 1.6718610
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8060098, upper bound: 1.8060946
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8060098, upper bound: 1.8060946
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8356677, upper bound: 1.8358020
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8356076, upper bound: 1.8358638
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8458158, upper bound: 1.8460647
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8457464, upper bound: 1.8462022
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8422709, upper bound: 1.8423601
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8422271, upper bound: 1.8424276
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8296177, upper bound: 1.8296560
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8296177, upper bound: 1.8296560
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.7887112, upper bound: 1.7887151
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.7887112, upper bound: 1.7887151
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8484133, upper bound: 1.8484596
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.90
Output dim: 6, lower bound: -1.8484131, upper bound: 1.8484752

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8804063, upper bound: 1.8798182
time: 3.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8802574, upper bound: 1.8799903
time: 2.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8780532, upper bound: 1.8777120
time: 3.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8780495, upper bound: 1.8777137
time: 2.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8773024, upper bound: 1.8770749
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8772531, upper bound: 1.8770749
time: 2.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8772695, upper bound: 1.8770136
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8772291, upper bound: 1.8770227
time: 2.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6102190, upper bound: 1.6102336
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.6102190, upper bound: 1.6102336
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7629818, upper bound: 1.7629221
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.7629818, upper bound: 1.7629221
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8384996, upper bound: 1.8384037
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8384980, upper bound: 1.8384037
time: 2.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8093817, upper bound: 1.8093664
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8093817, upper bound: 1.8093664
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8074083, upper bound: 1.8072489
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8074083, upper bound: 1.8072489
time: 2.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8171465, upper bound: 1.8169890
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8171465, upper bound: 1.8169890
time: 2.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454680, upper bound: 1.8452265
time: 2.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8454680, upper bound: 1.8452265
time: 2.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8354450, upper bound: 1.8351734
time: 2.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8353965, upper bound: 1.8352368
time: 2.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.4769118, 1.3269346, -0.4769118, 1.3269346, -1.8038464, 1.8038464
1: -0.5433673, 0.5461558, -0.5433673, 0.5461558, -1.0895231, 1.0895231
2: -0.6294971, 0.6858552, -0.6294971, 0.6858552, -1.3153522, 1.3153522
3: -0.4428055, 0.5047094, -0.4428055, 0.5047094, -0.9475149, 0.9475149
4: -0.5623835, 0.6495277, -0.5623835, 0.6495277, -1.2119112, 1.2119112
5: -0.6725667, 0.7972974, -0.6725667, 0.7972974, -1.4698641, 1.4698641
6: -0.6051830, 1.4891748, -0.6051830, 1.4891748, -2.0943580, 2.0943580
7: -0.6342677, 0.6814606, -0.6342677, 0.6814606, -1.3157284, 1.3157284
8: -0.6106737, 0.7169547, -0.6106737, 0.7169547, -1.3276284, 1.3276284
9: -0.4836978, 0.5420617, -0.4836978, 0.5420617, -1.0257595, 1.0257595

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 189
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 222
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8057544, upper bound: 1.8057558
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.8057544, upper bound: 1.8057558
time: 1.92 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 5.01 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8804063, upper bound: 1.8798182
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8802574, upper bound: 1.8799903
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8780532, upper bound: 1.8777120
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8780495, upper bound: 1.8777137
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8773024, upper bound: 1.8770749
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8772531, upper bound: 1.8770749
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8772695, upper bound: 1.8770136
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8772291, upper bound: 1.8770227
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.6102190, upper bound: 1.6102336
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.6102190, upper bound: 1.6102336
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.7629818, upper bound: 1.7629221
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.7629818, upper bound: 1.7629221
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8384996, upper bound: 1.8384037
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8384980, upper bound: 1.8384037
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8093817, upper bound: 1.8093664
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8093817, upper bound: 1.8093664
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8074083, upper bound: 1.8072489
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8074083, upper bound: 1.8072489
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8171465, upper bound: 1.8169890
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8171465, upper bound: 1.8169890
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8454680, upper bound: 1.8452265
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8454680, upper bound: 1.8452265
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8354450, upper bound: 1.8351734
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8353965, upper bound: 1.8352368
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8057544, upper bound: 1.8057558
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 5.01
Output dim: 6, lower bound: -1.8057544, upper bound: 1.8057558
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.01
Output dim: 6, lower bound: -1.8495774, upper bound: 1.8495581
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.01
Output dim: 6, lower bound: -1.8425025, upper bound: 1.8425272
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.01
Output dim: 6, lower bound: -1.8422916, upper bound: 1.8427101
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.01
Output dim: 6, lower bound: -1.8458158, upper bound: 1.8460647
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.01
Output dim: 6, lower bound: -1.8457464, upper bound: 1.8462022
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.01
Output dim: 6, lower bound: -1.8422709, upper bound: 1.8423601
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.01
Output dim: 6, lower bound: -1.8422271, upper bound: 1.8424276
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 5.01
Output dim: 6, lower bound: -1.8484133, upper bound: 1.8484596
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 5.01
Output dim: 6, lower bound: -1.8484131, upper bound: 1.8484752
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=2.094357967376709
rel_dist={6: [-1.912201525114949, 1.9122015106221726]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1244.94 seconds
